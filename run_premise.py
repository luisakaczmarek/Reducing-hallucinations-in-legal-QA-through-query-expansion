"""
run_premise.py
--------------
Condition C2: premise rejection for fake_* tasks only.
Samples 500 rows per task from local dataset.csv (same SEED=42 as run_baselines.py),
expands each query with a premise-rejection template, then runs against all models.

Only processes: fake_case_existence, fake_dissent, fake_year_overruled
Total: ~1,500 rows per model.

Requires:
    pip install openai groq google-genai tqdm python-dotenv pandas

.env keys expected:
    GROQ_API_KEY=...
    OPENAI_API_KEY=...
    GOOGLE_API_KEY=...

Output files (resumable):
    results/premise_groq.jsonl
    results/premise_gpt4omini.jsonl
    results/premise_gemini.jsonl
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH = "dataset.csv"
SAMPLE_PER_TASK = 500
SEED = 42
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

TASKS = [
    "fake_case_existence",
    "fake_dissent",
    "fake_year_overruled",
]

MODELS = {
    "groq": {
        "output_file": OUTPUT_DIR / "premise_groq.jsonl",
        "model_id": "llama-3.3-70b-versatile",
        "rate_limit_sleep": 2.1,
    },
    "gpt4omini": {
        "output_file": OUTPUT_DIR / "premise_gpt4omini.jsonl",
        "model_id": "gpt-4o-mini",
        "rate_limit_sleep": 0,
    },
    "gemini": {
        "output_file": OUTPUT_DIR / "premise_gemini.jsonl",
        "model_id": "gemini-2.5-flash",
        "rate_limit_sleep": 4.1,
    },
}

ALL_MODEL_KEYS = list(MODELS.keys())

SYSTEM_PROMPT = (
    "You are a legal research assistant. "
    "Answer the question directly and concisely. "
    "Do not hedge or refuse unless genuinely uncertain."
)

# ── Expansions ────────────────────────────────────────────────────────────────
EXPANSIONS = {
    "fake_case_existence": (
        "The following question asks whether a case exists.\n"
        "Examine the case name critically — it may be fabricated.\n"
        "If you have no reliable knowledge of this case in any federal court\n"
        "database, say it does not exist rather than confirming it.\n\n"
        "{query}"
    ),
    "fake_dissent": (
        "The following question contains an assertion about a dissenting opinion.\n"
        "This assertion may be false. Verify whether this dissent actually exists\n"
        "before accepting the premise. If you cannot confirm it, reject the premise\n"
        "explicitly.\n\n"
        "{query}"
    ),
    "fake_year_overruled": (
        "The following question asserts a case was overruled in a specific year.\n"
        "This assertion may be false. Before answering, consider:\n"
        "(1) was this case overruled at all?\n"
        "(2) is the stated year plausible?\n"
        "If the premise appears incorrect, say so explicitly.\n\n"
        "{query}"
    ),
}


def expand(query: str, task: str) -> str:
    if task not in EXPANSIONS:
        raise ValueError(f"No expansion template defined for task: {task}")
    return EXPANSIONS[task].format(query=query)


# ── Load & sample ─────────────────────────────────────────────────────────────
def load_sample() -> pd.DataFrame:
    print(f"Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    df = df[df["task"].isin(TASKS)].copy()
    print(f"\nRows after task filter: {len(df)}")
    print("Counts per task:")
    print(df["task"].value_counts().to_string())

    sampled = pd.concat(
        [grp.sample(min(SAMPLE_PER_TASK, len(grp)), random_state=SEED)
         for _, grp in df.groupby("task")]
    ).reset_index(drop=True)
    print(f"\nSampled {len(sampled)} rows total ({SAMPLE_PER_TASK}/task max)")
    return sampled


# ── Resume helpers ────────────────────────────────────────────────────────────
def count_completed(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_completed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            try:
                ids.add(str(json.loads(line)["row_id"]))
            except Exception:
                pass
    return ids


def append_result(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── API callers ───────────────────────────────────────────────────────────────
def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_groq(query: str, model_id: str, client) -> tuple[str, float]:
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=512,
    )
    text = _strip_think_tags(response.choices[0].message.content)
    return text, time.time() - start


def call_openai(query: str, model_id: str, client) -> tuple[str, float]:
    start = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip(), time.time() - start


def call_gemini(query: str, model_id: str, client) -> tuple[str, float]:
    start = time.time()
    response = client.models.generate_content(
        model=model_id,
        contents=f"{SYSTEM_PROMPT}\n\n{query}",
    )
    return response.text.strip(), time.time() - start


# ── Per-model runner ──────────────────────────────────────────────────────────
def run_model(model_key: str, df: pd.DataFrame):
    cfg = MODELS[model_key]
    out_path = cfg["output_file"]
    model_id = cfg["model_id"]
    sleep_s = cfg["rate_limit_sleep"]

    n_done = count_completed(out_path)
    expected = len(df)
    if n_done >= expected:
        print(f"\n{'='*60}")
        print(f"Model: {model_key} ({model_id})")
        print(f"SKIP — already complete ({n_done}/{expected} rows in {out_path})")
        return

    completed = load_completed_ids(out_path)
    print(f"\n{'='*60}")
    print(f"Model: {model_key} ({model_id})")
    print(f"Already completed: {len(completed)}")
    print(f"Output: {out_path}")

    if model_key == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        caller = call_groq
    elif model_key == "gpt4omini":
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        caller = call_openai
    elif model_key == "gemini":
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        caller = call_gemini

    todo = df[~df["id"].astype(str).isin(completed)].copy()
    print(f"Remaining: {len(todo)} rows\n")

    errors = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=model_key):
        row_id = str(row["id"])
        query = str(row["query"])
        task = row["task"]
        expanded_query = expand(query, task)

        for attempt in range(4):
            try:
                response_text, latency = caller(expanded_query, model_id, client)
                break
            except Exception as e:
                wait = 2 ** attempt
                tqdm.write(f"[id {row_id}] Error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            tqdm.write(f"[id {row_id}] FAILED after 4 attempts, skipping.")
            errors += 1
            continue

        record = {
            "row_id": row_id,
            "task": task,
            "query": query,
            "expanded_query": expanded_query,
            "true_answer": row.get("example_correct_answer", None),
            "court_level": row.get("court_level", None),
            "court_slug": row.get("court_slug", None),
            "model": model_id,
            "model_key": model_key,
            "condition": "premise",
            "response": response_text,
            "latency_s": round(latency, 3),
        }
        append_result(out_path, record)

        if sleep_s > 0:
            time.sleep(sleep_s)

    print(f"\nDone. Errors: {errors}")
    print(f"Total rows written: {count_completed(out_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=ALL_MODEL_KEYS + ["all"],
        default="all",
        help="Which model to run (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded query for one row per task without calling APIs",
    )
    args = parser.parse_args()

    df_sample = load_sample()

    if args.dry_run:
        print("\n--- DRY RUN: expanded query for first row per task ---")
        for task, grp in df_sample.groupby("task"):
            raw = str(grp.iloc[0]["query"])
            expanded = expand(raw, task)
            print(f"\n[{task}]\n{expanded}")
        print(f"\nModels configured: {ALL_MODEL_KEYS}")
        print("Re-run without --dry-run to call APIs.")
    else:
        models_to_run = ALL_MODEL_KEYS if args.model == "all" else [args.model]
        for m in models_to_run:
            run_model(m, df_sample)
        print("\nAll done. Results in:")
        for m in models_to_run:
            print(f"  {MODELS[m]['output_file']}")

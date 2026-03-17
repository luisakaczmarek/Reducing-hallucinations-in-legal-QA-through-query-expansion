#!/usr/bin/env python3
"""
run_premise.py — C2 premise rejection condition

Processes only the 3 fake_* tasks. Each task's expansion explicitly names
the false premise embedded in the question and requires independent
verification before answering.

Expansions are loaded from prompt_expansions.json.
Merge pattern:  expanded_query = expansion + "\\n\\n" + query

Usage:
    python run_premise.py --model groq
    python run_premise.py --model all
    python run_premise.py --model gemini --dry-run
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# ── Constants ──────────────────────────────────────────────────────────────

SEED             = 42
SAMPLES_PER_TASK = 500
CONDITION        = "premise"
OUTPUT_PREFIX    = "premise"

# Only the three false-premise tasks are evaluated in this condition
TASKS = [
    "fake_case_existence",
    "fake_dissent",
    "fake_year_overruled",
]

# Provider config: model ID and inter-request sleep to respect rate limits
MODEL_CONFIGS = {
    "groq":      {"model_id": "llama-3.3-70b-versatile", "sleep_s": 2.1},
    "gpt4omini": {"model_id": "gpt-4o-mini",             "sleep_s": 0.0},
    # "gemini":    {"model_id": "gemini-2.5-flash",         "sleep_s": 4.1},
}

# ── Data loading & sampling ────────────────────────────────────────────────

def load_sample(csv_path: str) -> pd.DataFrame:
    """
    Load dataset.csv and return a stratified sample of SAMPLES_PER_TASK rows
    per task, using SEED for reproducibility.

    Only the 3 fake_* tasks are sampled — TASKS is intentionally restricted
    compared to run_baselines.py and run_structured.py.

    Uses pd.concat over a list comprehension — groupby().apply() with a
    key column is not supported in pandas 3.0+.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # Guard against stray embedded header rows (task == "task")
    df = df[df["task"].isin(TASKS)]
    sampled = pd.concat([
        g.sample(n=min(SAMPLES_PER_TASK, len(g)), random_state=SEED)
        for _, g in df.groupby("task")
    ]).reset_index(drop=True)
    return sampled

# ── Prompt expansions ──────────────────────────────────────────────────────

def load_expansions(json_path: str) -> dict:
    """Load prompt_expansions.json and return the parsed dict."""
    with open(json_path) as f:
        return json.load(f)

# ── Resumability ───────────────────────────────────────────────────────────

def load_done_ids(output_path: str) -> set:
    """
    Parse an existing JSONL output file and return the set of row_ids
    already recorded. Allows safe restart after interruption.
    """
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["row_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done

# ── API clients ────────────────────────────────────────────────────────────

def build_clients() -> dict:
    """Initialise all three provider clients from .env keys."""
    load_dotenv()
    from groq import Groq
    from openai import OpenAI
    from google import genai

    return {
        "groq":      Groq(api_key=os.environ["GROQ_API_KEY"]),
        "gpt4omini": OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        "gemini":    genai.Client(api_key=os.environ["GOOGLE_API_KEY"]),
    }

# ── Retry wrapper ──────────────────────────────────────────────────────────

def call_with_retry(call_fn, max_attempts: int = 3):
    """
    Call call_fn up to max_attempts times.
    Waits 2s, 4s before the 2nd and 3rd attempts (exponential backoff).
    Returns None and prints a stderr warning if all attempts fail.
    """
    for attempt in range(max_attempts):
        try:
            return call_fn()
        except Exception as e:
            wait = 2 ** (attempt + 1)   # 2s → 4s → (fail)
            if attempt < max_attempts - 1:
                print(
                    f"Warning: attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(
                    f"Warning: all {max_attempts} attempts failed. Last error: {e}",
                    file=sys.stderr,
                )
    return None

# ── Model call ─────────────────────────────────────────────────────────────

def query_model(model_key: str, prompt: str, clients: dict) -> tuple[str, float | None]:
    """
    Send prompt to the specified model and return (response_text, latency_s).
    Returns ("ERROR", None) if all retry attempts fail.
    """
    from google.genai import types as genai_types

    config  = MODEL_CONFIGS[model_key]
    t_start = time.time()

    if model_key == "groq":
        def call():
            resp = clients["groq"].chat.completions.create(
                model=config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content

    elif model_key == "gpt4omini":
        def call():
            resp = clients["gpt4omini"].chat.completions.create(
                model=config["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content

    elif model_key == "gemini":
        def call():
            resp = clients["gemini"].models.generate_content(
                model=config["model_id"],
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0),
            )
            return resp.text

    result  = call_with_retry(call)
    latency = round(time.time() - t_start, 3)

    if result is None:
        return "ERROR", None
    return result, latency

# ── Query expansion ────────────────────────────────────────────────────────

def get_expanded_query(task: str, query: str, expansions: dict) -> str:
    """
    Premise rejection condition: prepend the task-specific premise-rejection
    expansion template from prompt_expansions.json.
    Merge pattern: expansions[task]["expansion"] + "\\n\\n" + query
    """
    expansion = expansions[task]["expansion"]
    return expansion + "\n\n" + query

# ── Main run loop ──────────────────────────────────────────────────────────

def run(model_key: str, df: pd.DataFrame, expansions: dict, dry_run: bool = False) -> None:
    """Process all sampled rows for a single model key."""
    output_path = f"results/{OUTPUT_PREFIX}_{model_key}.jsonl"
    os.makedirs("results", exist_ok=True)

    config = MODEL_CONFIGS[model_key]

    # ── Dry run: print one expanded query per task, then exit ──────────────
    if dry_run:
        print(f"\n=== Dry run: condition={CONDITION}  model={model_key} ===")
        for task in TASKS:
            task_rows = df[df["task"] == task]
            if task_rows.empty:
                continue
            row = task_rows.iloc[0]
            eq  = get_expanded_query(task, str(row["query"]), expansions)
            print(f"\n--- {task} ---\n{eq}")
        return

    done_ids = load_done_ids(output_path)
    clients  = build_clients()

    task_stats = {t: {"processed": 0, "skipped": 0} for t in TASKS}
    rows       = list(df.itertuples(index=False))

    with open(output_path, "a") as out_f:
        with tqdm(total=len(rows), desc=f"{model_key}/{CONDITION}", unit="row") as pbar:
            for row in rows:
                task   = row.task
                row_id = str(row.id)
                query  = str(row.query)

                pbar.set_postfix(task=task)

                # Skip rows already in the output file (resumability)
                if row_id in done_ids:
                    task_stats[task]["skipped"] += 1
                    pbar.update(1)
                    continue

                expanded_query    = get_expanded_query(task, query, expansions)
                response, latency = query_model(model_key, expanded_query, clients)

                record = {
                    "row_id":         row_id,
                    "task":           task,
                    "query":          query,
                    "expanded_query": expanded_query,
                    "true_answer":    str(row.example_correct_answer),
                    "court_level":    str(row.court_level),
                    "court_slug":     str(row.court_slug),
                    "model":          config["model_id"],
                    "model_key":      model_key,
                    "condition":      CONDITION,
                    "response":       response,
                    "latency_s":      latency,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                done_ids.add(row_id)
                task_stats[task]["processed"] += 1
                pbar.update(1)

                # Rate-limit sleep between requests
                if config["sleep_s"] > 0:
                    time.sleep(config["sleep_s"])

    # Per-task completion summary
    print(f"\n=== Summary: {CONDITION} / {model_key} ===")
    for task in TASKS:
        s = task_stats[task]
        print(f"  {task:<25} processed={s['processed']}  skipped={s['skipped']}")

# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run C2 premise rejection condition — fake_* tasks only, "
            "with explicit false-premise rejection expansion."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["groq", "gpt4omini", "gemini", "all"],
        help="Which model(s) to run. 'all' runs groq → gpt4omini → gemini sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one expanded query per task and exit without calling any API.",
    )
    args = parser.parse_args()

    df         = load_sample("legal_hallucination_data/dataset.csv")
    expansions = load_expansions("prompt_expansions.json")
    models     = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    for model_key in models:
        run(model_key, df, expansions, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

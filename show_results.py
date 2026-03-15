"""
show_results.py
---------------
Displays baseline run results from results/baseline_*.jsonl.
Works with partial or complete files.

Usage:
    python show_results.py                  # all available models
    python show_results.py --model groq
    python show_results.py --samples 3      # show N sample responses per task
    python show_results.py --no-samples     # skip samples
"""

import argparse
import json
import textwrap
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
SAMPLE_PER_TASK = 500  # target per task (from run_baselines.py)

TASK_ORDER = [
    "affirm_reverse", "case_existence", "citation_retrieval", "cited_precedent",
    "court_id", "fake_case_existence", "fake_dissent", "majority_author",
    "quotation", "fake_year_overruled", "year_overruled",
]

MODEL_FILES = {
    "groq":       RESULTS_DIR / "baseline_groq.jsonl",
    "gpt4omini":  RESULTS_DIR / "baseline_gpt4omini.jsonl",
}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {i}: {e}")
    return records


# ── Preliminary scoring ───────────────────────────────────────────────────────
def quick_match(response: str, true_answer: str) -> bool:
    """
    Case-insensitive substring check: does the response contain the true answer?
    This is a rough proxy — proper scoring (Dahl et al.) needs a dedicated script.
    For short answers (affirm/reverse, yes/no, a year) this is reasonably accurate.
    For long answers (quotations, citations) it's looser.
    """
    if not true_answer or str(true_answer).strip().lower() in ("", "nan", "none"):
        return False
    return str(true_answer).strip().lower() in response.strip().lower()


# ── Display helpers ───────────────────────────────────────────────────────────
SEP  = "─" * 72
SEP2 = "═" * 72

def hr(char="─"):
    print(char * 72)

def section(title: str):
    print()
    print(SEP2)
    print(f"  {title}")
    print(SEP2)

def wrap(text: str, width=68, indent="    ") -> str:
    return textwrap.fill(str(text), width=width, initial_indent=indent,
                         subsequent_indent=indent)


# ── Main display ──────────────────────────────────────────────────────────────
def display_model(model_key: str, path: Path, n_samples: int):
    print()
    print(SEP2)
    print(f"  MODEL: {model_key.upper()}  ({path.name})")
    print(SEP2)

    records = load_jsonl(path)
    if not records:
        print("  No records found.")
        return

    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    total = len(records)
    model_id = records[0]["model"]
    avg_latency = sum(r["latency_s"] for r in records) / total

    print(f"\n  Model ID : {model_id}")
    print(f"  Records  : {total:,}  (target: {len(TASK_ORDER) * SAMPLE_PER_TASK:,})")
    print(f"  Avg latency: {avg_latency:.2f}s/call")

    # ── Progress table ────────────────────────────────────────────────────────
    section("PROGRESS BY TASK")
    print(f"  {'Task':<25} {'Done':>6} {'Target':>7} {'Progress':>10}  {'~Match%':>8}")
    print(f"  {'─'*25} {'─'*6} {'─'*7} {'─'*10}  {'─'*8}")

    total_match = 0
    total_scored = 0

    tasks_in_order = [t for t in TASK_ORDER if t in by_task]
    tasks_in_order += [t for t in by_task if t not in TASK_ORDER]

    for task in tasks_in_order:
        rows = by_task[task]
        done = len(rows)
        bar_len = 12
        filled = round(bar_len * done / SAMPLE_PER_TASK)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = done / SAMPLE_PER_TASK * 100

        # Quick match rate
        scoreable = [r for r in rows if r.get("true_answer") not in (None, "", "nan")]
        if scoreable:
            matches = sum(quick_match(r["response"], r["true_answer"]) for r in scoreable)
            match_pct = matches / len(scoreable) * 100
            match_str = f"{match_pct:6.1f}%"
            total_match += matches
            total_scored += len(scoreable)
        else:
            match_str = "    N/A"

        print(f"  {task:<25} {done:>6,} {SAMPLE_PER_TASK:>7,} {bar} {pct:4.0f}%  {match_str}")

    print(f"  {'─'*25} {'─'*6}")
    print(f"  {'TOTAL':<25} {total:>6,} {len(TASK_ORDER)*SAMPLE_PER_TASK:>7,}")
    if total_scored:
        overall_match = total_match / total_scored * 100
        print(f"\n  Overall ~match rate: {overall_match:.1f}%  "
              f"(substring match vs true_answer — rough proxy only)")

    # ── Latency by task ───────────────────────────────────────────────────────
    section("LATENCY BY TASK  (seconds/call)")
    print(f"  {'Task':<25} {'N':>5}  {'Mean':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'─'*25} {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}")
    for task in tasks_in_order:
        rows = by_task[task]
        lats = [r["latency_s"] for r in rows]
        print(f"  {task:<25} {len(lats):>5,}  {sum(lats)/len(lats):>6.2f}  "
              f"{min(lats):>6.2f}  {max(lats):>6.2f}")

    # ── Sample responses ──────────────────────────────────────────────────────
    if n_samples > 0:
        section(f"SAMPLE RESPONSES  ({n_samples} per task)")
        for task in tasks_in_order:
            rows = by_task[task]
            print(f"\n  ┌─ {task.upper()} {'─'*(60-len(task))}")
            for row in rows[:n_samples]:
                matched = quick_match(row["response"], row.get("true_answer", ""))
                status = "✓" if matched else "✗"
                print(f"  │  [{status}] true: {str(row.get('true_answer','?'))[:60]}")
                print(wrap(f"Q: {row['query'][:200]}", indent="  │      "))
                print(wrap(f"A: {row['response'][:300]}", indent="  │      "))
                print(f"  │")
            print(f"  └{'─'*67}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["groq", "gpt4omini"], default=None,
                        help="Which model to show (default: all available)")
    parser.add_argument("--samples", type=int, default=2,
                        help="Sample responses per task (default: 2, 0 to disable)")
    parser.add_argument("--no-samples", action="store_true",
                        help="Disable sample display")
    args = parser.parse_args()

    n_samples = 0 if args.no_samples else args.samples

    models_to_show = [args.model] if args.model else list(MODEL_FILES.keys())

    found_any = False
    for model_key in models_to_show:
        path = MODEL_FILES[model_key]
        if not path.exists():
            print(f"\n[{model_key}] No file at {path} — skipping.")
            continue
        found_any = True
        display_model(model_key, path, n_samples)

    if not found_any:
        print("No result files found in results/. Run run_baselines.py first.")
    else:
        print()
        hr("─")
        print("  NOTE: '~Match%' is a rough substring match against true_answer.")
        print("  Final hallucination scoring (Dahl et al.) needs a separate script.")
        hr("─")


if __name__ == "__main__":
    main()

# Reducing Hallucinations in Legal QA Through Query Expansion

Testing whether **task-specific prompt-level interventions** reduce LLM hallucination rates on legal tasks, using the [Dahl et al. (2024)](https://academic.oup.com/jla/article/16/1/64/7590670) benchmark (`reglab/legal_hallucinations`).

---

## Research Design

### The Question
Can fixed, task-specific prompt templates applied at inference time — without any retrieval infrastructure — reduce LLM hallucination rates on legal QA tasks?

### Three Experimental Conditions

| # | Label | Description | Script |
|---|-------|-------------|--------|
| C0 | **Baseline** | Raw queries sent as-is | `run_baselines.py` |
| C1 | **Structured Reasoning** | Task-specific scaffolding prepended to each query (all 11 tasks) | `run_structured.py` |
| C2 | **Premise Rejection** | Explicit false-premise rejection instruction (`fake_*` tasks only) | `run_premise.py` |

### Key Design Decisions

- **No generative expansion step** — templates are fixed strings applied inline at inference time. This differs from Zheng et al. (2025), whose generative reasoning rollout was designed for long Bar Exam fact patterns — not the 5–7 word metadata queries in this dataset.
- **Paraphrase and CoT rejected** — queries are too short for paraphrasing to add value; chain-of-thought encourages confabulation on pure metadata retrieval tasks where the answer is a memorised fact, not a derivable one.
- **Zero-shot only** — Dahl et al. found no substantial difference between zero-shot and few-shot; one condition is sufficient.
- **Temperature = 0** throughout for determinism.

---

## Dataset

**Source**: `reglab/legal_hallucinations` (Dahl et al., 2024), `df_main` split  
**Local file**: `dataset.csv`

### Tasks (11 total)

| Task | Complexity | N rows |
|------|-----------|--------|
| `case_existence` | Low | 108,344 |
| `fake_case_existence` | Low (false premise) | 10,736 |
| `court_id` | Low | 108,344 |
| `citation_retrieval` | Low | 108,344 |
| `majority_author` | Low | 108,346 |
| `affirm_reverse` | Moderate | 72,256 |
| `cited_precedent` | Moderate | 108,344 |
| `quotation` | Moderate | 108,344 |
| `year_overruled` | Moderate | 1,776 |
| `fake_dissent` | Moderate (false premise) | 7,160 |
| `fake_year_overruled` | Moderate (false premise) | 3,612 |

**Sampling**: 500 rows per task, stratified random sample, `SEED=42` — identical across all scripts.  
- C0 and C1 process all 11 tasks → ~5,500 rows per model  
- C2 processes only the 3 `fake_*` tasks → ~1,500 rows per model

### Key columns
- `query` — sent to the LLM (with or without expansion)
- `example_correct_answer` — ground truth for scoring
- `correctness_score` and `hallucination` — Dahl et al.'s original runs; **not used as ground truth for new responses**
- `case_source == 'fake'` — identifies fabricated case rows

---

## Models

| Key | Model ID | Provider |
|-----|----------|----------|
| `groq` | `llama-3.3-70b-versatile` | Groq API |
| `gpt4omini` | `gpt-4o-mini` | OpenAI API |
| `gemini` | `gemini-2.5-flash` | Google AI Studio |

---

## Query Expansion Templates

Templates are applied as `template.format(query=query)` — the original query is always preserved at the end.

### C1 — Structured Reasoning (`run_structured.py`, all 11 tasks)

Each template frames the task type, provides structured reasoning cues, and instructs the model to acknowledge uncertainty rather than confabulate.

**`case_existence`**
```
You are answering a question about whether a federal court case exists.
Only confirm existence if you have reliable knowledge of this specific case
from a recognized legal database. If uncertain, say so explicitly.

{query}
```

**`court_id`**
```
You are identifying which federal court decided a case. Consider:
(1) SCOTUS, Circuit Court, or District Court level;
(2) geographic jurisdiction implied by the case name;
(3) the case's approximate era.
Answer with the court name only.

{query}
```

**`citation_retrieval`**
```
You are retrieving a case citation in Bluebook format. Consider:
(1) SCOTUS citations use U.S. Reports (e.g. 410 U.S. 113);
(2) Circuit citations use F.2d or F.3d;
(3) District citations use F.Supp. or F.Supp.2d.
If you are not certain of the exact volume and page, say so rather than guessing.

{query}
```

**`majority_author`**
```
You are identifying who wrote the majority opinion. Consider:
(1) the court level and approximate year of the decision;
(2) which judges or justices were active on that court at that time;
(3) what you reliably know about authorship of this specific case.
Answer with the judge's full name only.

{query}
```

**`affirm_reverse`**
```
You are determining whether an appellate court affirmed or reversed the
lower court. Consider the case's legal issue and the court's known
jurisprudence. Answer only with "affirmed" or "reversed".

{query}
```

**`cited_precedent`**
```
You are identifying a case cited as authority in a judicial opinion.
Answer with a specific case name and citation. If you are not certain
which authority this case cited, say so rather than inventing a citation.

{query}
```

**`year_overruled`**
```
You are identifying the year a case was overruled. Only provide a year
if you have reliable knowledge that this case was in fact overruled and
know the year with confidence. If the case was not overruled or you are
uncertain, say so explicitly.

{query}
```

**`quotation`**
```
You are reproducing a direct quotation from a judicial opinion.
Only quote text you are confident appears verbatim in this case.
Do not paraphrase or reconstruct language. If uncertain, say so.

{query}
```

The three `fake_*` tasks in `run_structured.py` use the **C2 premise rejection templates** (see below).

---

### C2 — Premise Rejection (`run_premise.py`, `fake_*` tasks only)

These templates are also included in `run_structured.py` for the `fake_*` tasks. The key mechanism is alerting the model that the premise of the question may be false, and instructing it to reject rather than accept a fabricated case.

**`fake_case_existence`**
```
The following question asks whether a case exists.
Examine the case name critically — it may be fabricated.
If you have no reliable knowledge of this case in any federal court
database, say it does not exist rather than confirming it.

{query}
```

**`fake_dissent`**
```
The following question contains an assertion about a dissenting opinion.
This assertion may be false. Verify whether this dissent actually exists
before accepting the premise. If you cannot confirm it, reject the premise
explicitly.

{query}
```

**`fake_year_overruled`**
```
The following question asserts a case was overruled in a specific year.
This assertion may be false. Before answering, consider:
(1) was this case overruled at all?
(2) is the stated year plausible?
If the premise appears incorrect, say so explicitly.

{query}
```

---

## File Structure

```
.
├── dataset.csv                  # Dahl et al. benchmark (not committed — download separately)
├── run_baselines.py             # C0: raw queries
├── run_structured.py            # C1: structured reasoning templates
├── run_premise.py               # C2: premise rejection templates (fake_* only)
├── evaluate.py                  # Scoring: fuzzy match → hallucination rates
├── show_results.py              # Print/visualise results table
├── analysis.ipynb               # Analysis notebook with all figures
├── requirements.txt
├── CLAUDE_CODE_CONTEXT.md       # Internal dev context
└── results/
    ├── baseline_{model}.jsonl
    ├── structured_{model}.jsonl
    ├── premise_{model}.jsonl
    ├── scores/                  # Scored JSONL files (output of evaluate.py)
    ├── hallucination_rates.csv  # Summary table
    └── fig_*.png                # Output figures
```

---

## Output Format

Each run script writes one JSONL record per query:

```json
{
  "row_id": "...",
  "task": "case_existence",
  "query": "...",
  "expanded_query": "...",
  "true_answer": "...",
  "court_level": "...",
  "court_slug": "...",
  "model": "llama-3.3-70b-versatile",
  "model_key": "groq",
  "condition": "structured",
  "response": "...",
  "latency_s": 1.23
}
```

All output files are **resumable** — completed row IDs are tracked and skipped on restart.

---

## Evaluation

Scoring replicates Dahl et al. (2024):
- **Method**: fuzzy string match (`rapidfuzz`) between `response` and `example_correct_answer`
- **Hallucination threshold**: `correctness_score ≤ 72`
- **Refusal sentinel**: `score == -99` (counts as non-hallucination)
- **Primary metric**: hallucination rate per task × condition × model
- **Secondary metrics**: refusal rate, breakdown by court level

Run:
```bash
python evaluate.py
```

---

## Setup

```bash
pip install openai groq google-genai tqdm python-dotenv pandas rapidfuzz
```

Create a `.env` file:
```
GROQ_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

Download the dataset:
```python
from datasets import load_dataset
ds = load_dataset("reglab/legal_hallucinations", split="df_main")
ds.to_csv("dataset.csv", index=False)
```

### Running

```bash
# Dry run — print expanded queries without calling APIs
python run_structured.py --dry-run
python run_premise.py --dry-run

# Run a single model
python run_structured.py --model groq

# Run all models
python run_structured.py --model all
python run_premise.py --model all

# Evaluate
python evaluate.py
```

---

## Reference

Dahl, M., Magesh, V., Suzgun, M., & Ho, D. E. (2024). Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models. *Journal of Legal Analysis*, 16, 64–93.

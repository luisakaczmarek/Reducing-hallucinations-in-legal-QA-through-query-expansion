# Claude Code Context: Query Expansion Hallucination Study

## Project overview
Testing whether **task-specific query expansion** reduces LLM hallucination rates on legal tasks,
using the `reglab/legal_hallucinations` benchmark dataset (Dahl et al., 2024).

---

## Research design

### 3 experimental conditions, same models, same rows
1. **Baseline (C0)** — raw queries as-is from dataset → `run_baselines.py` ← ALREADY RUNNING
2. **Structured reasoning (C1)** — task-specific scaffolding prepended to query → `run_structured.py`
3. **Premise rejection (C2)** — explicit false-premise rejection instruction, `fake_*` tasks only → `run_premise.py`

### Key design decisions
- **No separate expansion generation step** — templates are fixed strings applied inline at inference time. This differs from Zheng et al. (2025) who used a generative rollout step; that made sense for long Bar Exam fact patterns, not for the 5–7 word queries in this dataset.
- **Paraphrase and CoT were considered and rejected**: queries are too short and simple for paraphrasing to add value; CoT encourages confabulation on pure metadata retrieval tasks where the answer is a memorized fact, not a derivable one.
- **Zero-shot only**: Dahl et al. found no substantial difference between zero-shot and few-shot; one condition is sufficient.
- Each script is independent — do not modify `run_baselines.py` while it is running.

---

## Dataset

- **File**: `dataset.csv` (df_main split of `reglab/legal_hallucinations`)
- **Columns**: `id`, `task`, `court_level`, `prompt_style`, `llm`, `temperature`,
  `case_source`, `court_slug`, `citation`, `year`, `query`, `llm_output`,
  `correctness_score`, `hallucination`, `example_correct_answer`
- `query` → send to LLM
- `example_correct_answer` → ground truth for scoring
- `correctness_score` and `hallucination` in CSV are Dahl et al.'s original runs — **do NOT use as ground truth for new responses**
- `case_source == 'fake'` identifies fake-case rows (used in `fake_*` tasks)

### Tasks (11 total, df_main)
```
affirm_reverse, case_existence, citation_retrieval, cited_precedent,
court_id, fake_case_existence, fake_dissent, majority_author,
quotation, fake_year_overruled, year_overruled
```

Task sizes:
- majority_author: 108,346
- case_existence: 108,344
- citation_retrieval: 108,344
- cited_precedent: 108,344
- court_id: 108,344
- quotation: 108,344
- affirm_reverse: 72,256
- fake_case_existence: 10,736
- fake_dissent: 7,160
- fake_year_overruled: 3,612
- year_overruled: 1,776

### Sampling
- **500 rows per task**, stratified random sample, **SEED=42** — must be identical across all scripts
- Total: ~5,500 rows per model per condition
- `run_premise.py` only processes the 3 fake tasks → ~1,500 rows

---

## Models
| Key | Model ID | Provider | Rate limit | Sleep |
|---|---|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | Groq API | ~30 req/min | 2.1s |
| `gpt4omini` | `gpt-4o-mini` | OpenAI API | — | none |
| `gemini` | `gemini-2.5-flash` | Google AI Studio | ~15 req/min | 4.1s |

All models run at **temperature=0**.

### .env keys required
```
GROQ_API_KEY=...
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
```

---

## Scripts (parallel structure — one per condition)

| Script | Condition | Output files |
|---|---|---|
| `run_baselines.py` | C0 baseline | `results/baseline_{model}.jsonl` |
| `run_structured.py` | C1 structured reasoning | `results/structured_{model}.jsonl` |
| `run_premise.py` | C2 premise rejection | `results/premise_{model}.jsonl` |

All scripts share:
- Same sampling logic (SEED=42, 500/task)
- Same model config, retry/backoff, rate-limit logic
- Same JSONL output format (see below)
- Resumable completion check keyed on output file + `id`
- `--dry-run` flag printing expanded query per task (not raw query)
- `--model` flag: `groq`, `gpt4omini`, `gemini`, `both`, `all`

Each JSONL record:
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

---

## Query expansion templates

### C1 — Structured Reasoning (run_structured.py, all 11 tasks)

Apply as: `template.format(query=query)` — original query always preserved at end.

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

**`fake_case_existence`** — use C2 template (same template in run_structured.py and run_premise.py)

**`fake_dissent`** — use C2 template

**`fake_year_overruled`** — use C2 template

---

### C2 — Premise Rejection (run_premise.py, fake_* tasks only)
Also included in run_structured.py for the fake_* tasks.

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

## Evaluation (not yet implemented — next step)

- **Script to write**: `evaluate.py`
- **Method**: replicate Dahl et al. — fuzzy string match between `response` and `example_correct_answer`
- **Threshold**: `correctness_score ≤ 72` = hallucinated; `-99` = refusal (counts as non-hallucination)
- **Primary metric**: hallucination rate per task per condition per model
- **Secondary**: breakdown by court level, refusal rate per condition
- Load all JSONL files, join on `row_id` across conditions, compute Δ hallucination rate (condition − baseline)

---

## Dependencies
```bash
pip install openai groq google-genai tqdm python-dotenv pandas
```

---

## Current status
- `run_baselines.py` — running (do not touch)
- `run_structured.py` — to be created
- `run_premise.py` — to be created
- `evaluate.py` — not yet started

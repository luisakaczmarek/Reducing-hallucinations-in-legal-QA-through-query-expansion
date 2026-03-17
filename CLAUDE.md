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
- **No separate expansion generation step** — templates are fixed strings loaded from `prompt_expansions.json` and applied inline at inference time. This differs from Zheng et al. (2025) who used a generative rollout step; that made sense for long Bar Exam fact patterns, not for the 5–7 word queries in this dataset.
- **Paraphrase rejected**: Zheng et al. (2025) confirmed paraphrasing either makes no difference or hurts performance on legal retrieval tasks. Excluded from all conditions.
- **CoT rejected**: CoT encourages confabulation on pure metadata retrieval tasks where the answer is a memorised fact, not a derivable one. Yu et al. (ACL 2023) showed IRAC-structured prompts outperform generic CoT on legal tasks; the tier structure below reflects this.
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

> Note: reference-free tasks (holding, posture, core_legal_question,
> factual_background, subsequent_history) are excluded — they belong
> to a separate split and are not evaluated in this study.

Task sizes:
- majority_author:      108,346
- case_existence:       108,344
- citation_retrieval:   108,344
- cited_precedent:      108,344
- court_id:             108,344
- quotation:            108,344
- affirm_reverse:        72,256
- fake_case_existence:   10,736
- fake_dissent:           7,160
- fake_year_overruled:    3,612
- year_overruled:         1,776

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

Templates are loaded from `prompt_expansions.json` at runtime.

**Merge pattern**: `expansion + "\n\n" + query` — original query is always preserved unchanged at the end.

```python
import json

with open("prompt_expansions.json") as f:
    expansions = json.load(f)

expanded_query = expansions[task]["expansion"] + "\n\n" + original_query
```

### Tier structure

Each tier targets a structurally distinct failure mode identified in the literature (Dahl et al., 2024; Feng et al., 2024; sycophancy literature):

**TIER_1** — Anti-sycophancy + epistemic permission
```
case_existence, fake_case_existence, court_id,
citation_retrieval, cited_precedent, majority_author
```
Failure mode: metadata confabulation — model generates plausible-sounding
metadata from pattern recognition (citation format, reporter abbreviation,
judge tenure) rather than actual recall. Expansion explicitly names and
prohibits each task-specific inference shortcut. Permits abstention
("unknown" / "no") — under Dahl's framework abstention = non-hallucination.

**TIER_2** — Stepwise memory-grounded decomposition
```
affirm_reverse, quotation, year_overruled,
fake_year_overruled, fake_dissent
```
Failure mode: reconstructive hallucination — model interpolates plausible
content from noisy parametric memory with no internal checkpoint to interrupt
reconstruction. Expansion forces stepwise recall before answer generation,
separating the recall check from the output (operationalising the CoVe
independence principle, Dhuliawala et al., ACL 2024). Prohibits task-specific
inference shortcuts (e.g. base-rate affirmance/reversal rates for
affirm_reverse; period-appropriate stylistic reconstruction for quotation).

### C1 — Structured Reasoning (run_structured.py, all 11 tasks)

All 11 tasks use their TIER_1 or TIER_2 expansion from `prompt_expansions.json`.
The 3 fake_* tasks use their C2 (premise rejection) expansion — see below.

### C2 — Premise Rejection (run_premise.py, fake_* tasks only)

Also applied to fake_* rows within run_structured.py.
`fake_case_existence`, `fake_year_overruled`, `fake_dissent` use their
dedicated premise-rejection expansions from `prompt_expansions.json`.

These tasks embed false premises (fabricated case, false overruling, fabricated
dissent). The C2 expansion explicitly names the false premise and requires
independent verification before answering — targeting counterfactual bias
as identified by Dahl et al. (2024).

### Design rationale summary

| Decision | Rationale |
|---|---|
| Templates name specific failure mechanism | Generic "be accurate" instructions do not interrupt confabulation pathways |
| Abstention permitted in all templates | Abstention = non-hallucination under Dahl's measurement framework |
| No paraphrasing | Confirmed to hurt or make no difference on legal tasks (Zheng et al., 2025) |
| No CoT | Encourages confabulation on metadata recall tasks; IRAC > CoT on legal tasks (Yu et al., ACL 2023) |
| C2 names the false premise explicitly | More effective at triggering premise rejection than generic accuracy instructions (sycophancy literature) |
| Fixed templates, not generated per-instance | Enables clean attribution of effects to prompt structure; ~600k queries make per-instance generation infeasible |

### Expected effect sizes by tier

These are predictions, not hedges — differential effect sizes are themselves a study contribution:

- **TIER_1**: strongest and most mechanistically predictable effects. Abstention permission directly and mechanically reduces measured hallucination rates on fake_* tasks under Dahl's own framework. Sycophancy interventions are most tractable at prompt level.
- **TIER_2**: moderate effects. Stepwise decomposition interrupts reconstruction but cannot substitute absent parametric knowledge.
- **fake_* tasks (C2)**: strongest overall — false premises are the most directly addressable failure mode at prompt level.
- **quotation**: lowest expected effect within TIER_2. Verbatim recall is structurally unreliable; the expansion redirects to abstention rather than producing correct quotations.

Note: Feng et al. (ACL 2024) establish that prompt-based abstention is the weakest abstention category overall (avg A-Acc 0.475), and drops to 33.3% on jurisprudence-adjacent tasks. This bounds realistic expectations — the study tests the minimum viable intervention under known-difficult conditions.

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

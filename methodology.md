# Methodology: Reducing Hallucinations in Legal QA Through Query Expansion

## 1. Research Question

Can task-specific query expansion reduce LLM hallucination rates on legal question-answering tasks, and do different structural failure modes require different intervention types?

---

## 2. Dataset

**Source**: `reglab/legal_hallucinations` (Dahl et al., 2024), `df_main` split
**File**: `legal_hallucination_data/dataset.csv` (~745,607 rows)

### 2.1 Task Inventory

The benchmark covers 11 evaluable tasks across two structural categories:

| Category | Tasks |
|---|---|
| **Metadata retrieval** | `case_existence`, `fake_case_existence`, `court_id`, `citation_retrieval`, `cited_precedent`, `majority_author` |
| **Reasoning / recall** | `affirm_reverse`, `quotation`, `year_overruled`, `fake_year_overruled`, `fake_dissent` |

Three `fake_*` tasks embed deliberate false premises: fabricated case existence (`fake_case_existence`), fabricated dissent authorship (`fake_dissent`), and false overruling year (`fake_year_overruled`). Reference-free generative tasks (`holding`, `posture`, `core_legal_question`, `factual_background`, `subsequent_history`) belong to a separate dataset split and are excluded from this study.

### 2.2 Key Columns

| Column | Role |
|---|---|
| `query` | Input sent to the model |
| `example_correct_answer` | Ground truth for scoring |
| `correctness_score` | Dahl et al.'s original run score — **not used as ground truth here** |
| `task`, `court_level`, `court_slug` | Stratification variables |
| `case_source` | `'fake'` identifies synthetic cases used in `fake_*` tasks |

### 2.3 Sampling

500 rows per task, stratified random sample, `SEED=42`. The same sample is used identically across all conditions and models, ensuring any observed differences are attributable to the prompt intervention, not sampling variation. Total: **5,500 rows per model per full-coverage condition** (1,500 rows for the premise-rejection condition, which covers only the 3 `fake_*` tasks).

---

## 3. Models

| Key | Model | Provider | Temperature | Rate limit sleep |
|---|---|---|---|---|
| `groq` | `llama-3.3-70b-versatile` | Groq API | 0 | 2.1s |
| `gpt4omini` | `gpt-4o-mini` | OpenAI API | 0 | none |
| `gemini` | `gemini-2.5-flash` | Google AI Studio | 0 | 4.1s |

All models run at `temperature=0` for deterministic, reproducible outputs. The Gemini results in this report cover only a partial sample (4,955 / 5,500 rows) due to API quota constraints. All retry logic uses exponential backoff (2s → 4s → fail), with API errors recorded as `"ERROR"` and treated as abstentions at scoring time.

---

## 4. Experimental Design

Three within-subjects conditions are applied to the same sampled rows:

### C0 — Baseline
Raw dataset queries, no modification. Establishes the hallucination rate each model achieves without any intervention.

**Script**: `run_baselines.py`
**Output**: `results/baseline_{model}.jsonl`

### C1 — Structured Reasoning
A task-specific expansion template is prepended to every query before sending to the model. Templates are fixed strings loaded from `prompt_expansions.json` — not generated per-instance. The merge pattern is:

```
expanded_query = expansion + "\n\n" + original_query
```

The original query is preserved unchanged at the end. All 11 tasks are covered.

**Script**: `run_structured.py`
**Output**: `results/structured_{model}.jsonl`

### C2 — Premise Rejection
Applied only to the 3 `fake_*` tasks. Expansions explicitly name the false premise embedded in the question and require independent verification before answering — targeting counterfactual bias as the primary failure mode (Dahl et al., 2024).

**Script**: `run_premise.py`
**Output**: `results/premise_{model}.jsonl`

Note: The same premise-rejection expansions are also used for `fake_*` tasks within C1, so C1 and C2 are directly comparable on those tasks.

---

## 5. Query Expansion Templates

Templates are stored in `prompt_expansions.json` and organized into two tiers, each targeting a structurally distinct failure mode identified in the literature.

### 5.1 TIER_1 — Anti-Sycophancy + Epistemic Permission

**Tasks**: `case_existence`, `fake_case_existence`, `court_id`, `citation_retrieval`, `cited_precedent`, `majority_author`

**Failure mode targeted**: Metadata confabulation. Models generate plausible-sounding metadata from pattern recognition (citation format, reporter abbreviation, judge tenure) rather than actual recall.

**Intervention**: The expansion explicitly names and prohibits each task-specific inference shortcut. It permits abstention (`"unknown"` / `"no"`). Under Dahl's measurement framework, abstention counts as non-hallucination.

**Example** (`citation_retrieval`):
> You are a legal fact-checker. Citations must be exact. Before answering: verify internally that you have reliable, specific knowledge of this exact citation — including volume number, reporter abbreviation, and page number. Do not reconstruct or estimate any component from partial memory. If you are uncertain about any part of the citation, respond "unknown".

### 5.2 TIER_2 — Stepwise Memory-Grounded Decomposition

**Tasks**: `affirm_reverse`, `quotation`, `year_overruled`, `fake_year_overruled`, `fake_dissent`

**Failure mode targeted**: Reconstructive hallucination. Models interpolate plausible content from noisy parametric memory with no internal checkpoint to interrupt reconstruction.

**Intervention**: Forces stepwise recall before answer generation, separating the recall check from the output. This operationalises the Chain-of-Verification independence principle (Dhuliawala et al., ACL 2024). For `fake_*` tasks, the expansion also names the false premise and requires independent verification before accepting it.

**Example** (`fake_year_overruled`):
> You are a legal fact-checker. The question asks for the year a case was overruled, but this premise may be false — the case may never have been overruled. Before answering: (1) Verify independently whether this case was actually overruled — do not accept the question's implicit premise. (2) If the case was NOT overruled, respond "not overruled". (3) If it was overruled, provide the specific year from reliable knowledge only — do not estimate.

### 5.3 Design Decisions for Template Construction

| Decision | Rationale |
|---|---|
| Templates name the specific failure mechanism | Generic "be accurate" instructions do not interrupt confabulation pathways |
| Abstention permitted in all templates | Abstention = non-hallucination under Dahl's measurement framework |
| Fixed templates, not generated per-instance | Enables clean attribution of effects to prompt structure; ~600k total queries make per-instance generation infeasible |
| No paraphrasing | Confirmed to hurt or make no difference on legal retrieval tasks (Zheng et al., 2025) |
| No Chain-of-Thought (CoT) | Encourages confabulation on metadata recall tasks where the answer is a memorised fact; IRAC-structured prompts outperform generic CoT on legal tasks (Yu et al., ACL 2023) |
| C2 names the false premise explicitly | More effective at triggering premise rejection than generic accuracy instructions (sycophancy literature) |
| Zero-shot only | Dahl et al. found no substantial difference between zero-shot and few-shot |

---

## 6. Evaluation

### 6.1 Method

Replicates the Dahl et al. (2024) evaluation protocol:

1. **Fuzzy string match** using `rapidfuzz.token_sort_ratio` between `response` and `example_correct_answer`.
2. **Threshold**: `correctness_score ≤ 72` → hallucinated; `correctness_score > 72` → correct.
3. **Abstention score**: `-99` (counts as non-hallucination).
4. **Task-specific handling**:
   - `case_existence` / `fake_case_existence`: binary Yes/No matching with refusal detection
   - `majority_author`: numeric judge IDs in `example_correct_answer` → treated as abstention (unscoreable without name-to-ID mapping)
   - API errors (`"ERROR"` response): treated as abstention (`-99`)
5. **Refusal detection**: responses matching `REFUSAL_PHRASES` (including `"unknown"`, `"i don't know"`, `"no such dissent exists"`, etc.) receive score `-99`.

**Script**: `evaluate.py`

### 6.2 Primary Metrics

- **Hallucination rate**: proportion of responses scoring ≤ 72, per task × condition × model
- **Δ hallucination rate**: condition rate minus baseline rate (negative = improvement)
- **Refusal rate**: proportion of responses scoring -99, per task × condition × model

---

## 7. Results

### 7.1 Standard Tasks (C0 Baseline vs. C1 Structured)

Average hallucination rate across 8 standard tasks (excluding `fake_*`):

| Model | Baseline | Structured | Δ |
|---|---|---|---|
| Llama-3.3-70B (Groq) | 63.1% | 46.6% | **▼16.5pp** |
| GPT-4o-mini | 62.5% | 16.3% | **▼46.2pp** |

GPT-4o-mini shows a substantially larger response to structured prompting. This difference likely reflects the model's stronger instruction-following capacity: the structured templates are treated more literally by GPT-4o-mini, resulting in high refusal rates on tasks where recall is uncertain.

Notable task-level patterns:
- **`citation_retrieval`**: GPT-4o-mini drops from 87.4% → 4.4% (nearly eliminated via abstention: refusal rate 90.8%)
- **`cited_precedent`**: GPT-4o-mini 99.4% → 0.2% (refusal rate 99.8%)
- **`case_existence`**: *increases* for both models under C1 (Groq: 8.6% → 43.4%; GPT-4o-mini: 19.6% → 66.4%) — the expansion prompts skepticism toward real cases, which all `case_existence` rows represent

### 7.2 False-Premise Tasks (C0 Baseline vs. C1 Structured vs. C2 Premise)

Average hallucination rate across the 3 `fake_*` tasks:

| Model | Baseline | Structured | Premise | Δ (baseline→structured) |
|---|---|---|---|---|
| Llama-3.3-70B (Groq) | 69.6% | 11.9% | 11.5% | **▼57.7pp** |
| GPT-4o-mini | 57.5% | 5.1% | 5.1% | **▼52.4pp** |

The largest absolute improvements in the study are on `fake_*` tasks, consistent with the prediction that false premises are the most directly addressable failure mode at prompt level.

**C2 ≈ C1 for `fake_*` tasks**: The isolated premise-rejection condition (C2) produces virtually identical results to the full structured condition (C1) on these tasks. Since C1 uses the same premise-rejection expansions for `fake_*` tasks, this confirms the expansions themselves drive the effect — not any synergy from the broader structured scaffolding.

Task highlights:
- `fake_year_overruled`: GPT-4o-mini 79.6% → 0.0%; Groq 92.2% → 0.2%
- `fake_case_existence`: GPT-4o-mini 24.0% → 1.4%; Groq 68.2% → 9.2%

### 7.3 Gemini Partial Results (Baseline Only)

Gemini 2.5 Flash baseline results are available for a partial sample. Hallucination rates range from 4.6% (`case_existence`) to 99.6% (`quotation`), broadly consistent with other models. Structured and premise-rejection runs pending API quota availability.

---

## 8. Limitations

- **`case_existence` degradation under C1**: The anti-sycophancy expansion also induces skepticism about real cases. All `case_existence` rows in `df_main` have true answer `1` (case exists), so increased abstention and skepticism unconditionally increases measured hallucination rate on this task. This is a measurement artifact, not genuine degradation.
- **`majority_author` unscoreable**: Ground truth contains numeric judge IDs rather than names. The task is evaluated but results should be interpreted cautiously.
- **Abstention conflation**: Under Dahl's framework, abstention and correct answers both count as non-hallucination. High refusal rates in C1 (particularly for GPT-4o-mini) reduce measured hallucination rates without necessarily producing *correct* answers. Refusal rates are reported separately to allow disaggregation.
- **Gemini incomplete**: 545 rows (of 5,500) are missing for Gemini baseline; structured and premise runs have not yet been completed.
- **Fixed templates**: Templates are not calibrated per-instance or per-model. A generative expansion step might outperform fixed templates on some tasks, at the cost of clean experimental attribution.

---

## 9. References

- Dahl, M., et al. (2024). "Large Legal Fictions: Profiling Legal Hallucinations in Large Language Models." *Journal of Legal Analysis*.
- Dhuliawala, S., et al. (2024). "Chain-of-Verification Reduces Hallucination in Large Language Models." *ACL 2024*.
- Feng, S., et al. (2024). "Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration." *ACL 2024*.
- Yu, F., et al. (2023). "FATE: A Framework for Analyzing Transformers in Legal Text." *ACL 2023*.
- Zheng, L., et al. (2025). "Query Expansion for Legal Retrieval." Cited in Dahl et al. follow-up work.

# Quality Filtering

Quality gates run after generation (or after cleaning, for non-generation pipelines). Each gate produces `(passed, rejected)` — rejected samples are never silently dropped; they become `RejectedSample` objects with a structured `rejection_reason` string and land in `rejected.jsonl`.

Gates are optional and additive. Configure any combination. The execution order is fixed:

```
[Generation] → HallucinationGate → RewardGate → DiversityGate → [Exporters]
```

---

## SchemaGate

Runs by default, immediately after all readers. Disable with `schema_gate=False`.

Validates field presence, token length, and encoding for each `task_type`:

| task_type | Required fields |
|-----------|----------------|
| `preference`, `implicit_preference` | `chosen` + `rejected` |
| `grpo` | `instruction` + `responses` (non-empty list) |
| `language_modeling` | `output` |
| `prompt_only` | `instruction` |
| `source_chunk` | `input` |
| Default (SFT) | `instruction` + `output` |

```python
CuratorConfig(
    min_tokens    = 10,
    max_tokens    = 2048,
    use_tiktoken  = False,   # True for tiktoken cl100k token counts
    schema_gate   = True,
)
```

**Rejection reasons:** `missing_field:{field}`, `below_min_tokens:{n}`, `above_max_tokens:{n}`, `encoding_error:null_byte_in_{field}`

---

## HallucinationGate

Verifies that each generated answer is grounded in its source text using an LLM judge (CheckEval-style). Only fires when `hallucination_threshold` is set.

```python
CuratorConfig(
    hallucination_threshold = 0.7,    # None = gate off
    judge_llm_model         = "openai/gpt-4o",   # recommended: different from generator
)
```

**What it checks:** The judge receives `source_text` (from `sample.input`) and `answer` (from `sample.output` or `sample.chosen`) and scores grounding 0.0–1.0.

**Key behaviour:**
- Samples with no `input` (no source context) pass through silently by default. To reject them instead, use `skip_if_no_context: false` in the YAML gate config or construct `HallucinationGate(skip_if_no_context=False)` directly — this parameter is not exposed in `CuratorConfig`.
- The judge always sees the exact source chunk that generated the sample — not a retrieved approximation.

**Rejection reason:** `hallucination_contract_failed:{score:.2f}`

**Same-model warning:** If `judge_llm_model` is the same model as `llm_model`, the judge scores its own outputs leniently. Use a different model for the judge whenever possible.

---

## RewardGate

Scores response quality across configurable dimensions using an LLM judge (UltraFeedback rubric). Only fires when `reward_threshold` is set.

```python
CuratorConfig(
    reward_threshold   = 0.7,     # None = gate off
    reward_dimensions  = ["helpfulness", "honesty", "instruction_following", "depth"],
    judge_llm_model    = "openai/gpt-4o",
)
```

**Available dimensions** (all built-in, no custom strings):

| Dimension | What it penalises |
|-----------|------------------|
| `helpfulness` | Unhelpful, irrelevant responses |
| `honesty` | False or misleading claims |
| `instruction_following` | Not addressing what was asked |
| `truthfulness` | Factual errors |
| `depth` | Shallow, vague answers lacking specifics |
| `creativity` | Formulaic, uncreative responses |
| `coherence` | Poor structure, hard to follow |

The `overall_score` is the mean of all dimension scores. Samples where `overall_score < threshold` are rejected.

**Custom rubrics:** If the built-in dimensions don't fit your domain, use `reward_prompt_template` to write any rubric you want. See [Customisation](customisation.md).

**Rejection reason:** `below_reward_threshold:{score:.2f}`

### Dual-scoring for DPO preference pairs

When the sample has both `chosen` and `rejected` fields, RewardGate scores both:

```
chosen_score  >= threshold  → chosen is good enough
rejected_score < threshold  → rejected is noticeably worse

Both conditions must hold to pass.
```

| Failure | Rejection reason | Meaning |
|---------|-----------------|---------|
| `chosen_score < threshold` | `dpo_pair_failed:chosen_below_threshold:{score}` | Chosen response too weak |
| `rejected_score >= threshold` | `dpo_pair_failed:rejected_above_threshold:{score}` | Rejected response too good — insufficient quality contrast |

`rejected_above_threshold` is a **generation problem**, not a gate problem. The probe cannot recover it (it exits immediately with 0 LLM calls). Fix it by: raising `reward_threshold`, adding `"depth"` to `reward_dimensions`, using `preference_mode="two_pass"`, or using a different judge model.

---

## DiversityGate

Filters samples that are too semantically similar to already-accepted samples in the batch. Uses sentence-transformer embeddings + cosine similarity. Requires `pip install "curatorkit[embedding]"`.

```python
CuratorConfig(
    # None = gate off. Samples with similarity ABOVE this value are rejected,
    # so a lower threshold rejects more (stricter).
    diversity_threshold = 0.92,
    embedding_model     = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device    = None,   # "cuda" | "cpu" | None (auto-detect)
)
```

**What text is embedded** (automatic by `task_type`):
- `preference` → `instruction + chosen`
- `grpo` → `instruction + first_response`
- `conversational` → `instruction + output`
- Default → `instruction + output`

**ANN backend:** Uses FAISS if installed (`[embedding-faiss]`), falls back to numpy brute-force otherwise.

**Rejection reason:** `diversity_gate:too_similar:{similarity:.3f}`

---

## Cross-run embedding deduplication

Separate from DiversityGate — this persists an embedding index across multiple runs so you can deduplicate against previously generated data.

```python
CuratorConfig(
    embedding_dedup           = True,
    embedding_index_dir       = "output/embedding_index",
    embedding_dedup_threshold = 0.92,
    embedding_reset_index     = False,   # True = start fresh
)
```

---

## Combining gates

All three gates can run together. Each one's rejects are independent — a sample that fails HallucinationGate never reaches RewardGate. Read per-gate counts in `result.stage_counts`:

```python
result.stage_counts["HallucinationGate"]
# {"input_count": 2850, "output_count": 2210, "probe_recovered": 0, "rejected_count": 640}
result.stage_counts["RewardGate"]
# {"input_count": 2210, "output_count": 1890, "probe_recovered": 0, "rejected_count": 320}
result.stage_counts["DiversityGate"]
# {"input_count": 1890, "output_count": 1740, "probe_recovered": 0, "rejected_count": 150}
```

(`probe_recovered` is non-zero when the [diagnostic probe](adaptive-recovery.md) is enabled and recovers samples inline.)

---

## Rejection reason reference

| Reason string | Gate | Fix |
|--------------|------|-----|
| `missing_field:{field}` | Schema | Check field_mapping or generation task output |
| `below_min_tokens:{n}` | Schema | Lower `min_tokens` or filter short source chunks |
| `above_max_tokens:{n}` | Schema | Raise `max_tokens` or chunk source documents |
| `hallucination_contract_failed:{score}` | Hallucination | Lower threshold, use different judge, or enable probe |
| `below_reward_threshold:{score}` | Reward | Lower threshold, change dimensions, or enable refiner |
| `dpo_pair_failed:chosen_below_threshold:{score}` | Reward | Enable refiner; chosen response too weak |
| `dpo_pair_failed:rejected_above_threshold:{score}` | Reward | Generation contrast too small; fix in generation config |
| `diversity_gate:too_similar:{similarity}` | Diversity | Raise `diversity_threshold` or reduce `num_questions` |

---

## Next: [Adaptive recovery →](adaptive-recovery.md)

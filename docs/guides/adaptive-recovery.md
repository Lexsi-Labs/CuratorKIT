# Adaptive Recovery

Not all gate rejections are real failures. Some happen because the LLM picked the wrong temperature, the prompt framing was slightly off, or the score landed just below the threshold. The recovery system diagnoses *why* a sample was rejected and attempts a targeted repair before declaring it lost.

Two mechanisms work at different points in the pipeline:

| Mechanism | When it runs | What it does |
|-----------|-------------|-------------|
| **Inline probe** | Immediately after each gate's rejects, before the next gate | Diagnoses the failure mode, retries generation at different temperatures or with different prompts, re-evaluates |
| **Reward refiner** | Post-pipeline, after all gates | Rewrites the answer targeting the weakest quality dimension, re-evaluates with RewardGate |

Enable both with:
```python
CuratorConfig(
    enable_diagnostic_probe = True,
    enable_reward_refiner   = True,
)
```

---

## How inline recovery works

The pipeline runner checks for an attached probe after every gate. If one exists and there are rejected samples, it runs `probe.diagnose_batch(rejected)` before passing samples to the next stage:

```
Gate N
  ├─► passed
  └─► rejected ──► probe.diagnose_batch()
                     ├─► recovered  ──► merged with passed ──► Gate N+1
                     └─► still rejected ──► final rejected list
```

This means: if a probe is attached to both HallucinationGate and RewardGate, HallucinationGate probe recovery fires before RewardGate sees any samples.

---

## Inline probe routing

The probe uses the gate's score to decide which path to try first:

```
Score >= probe_score_split (default 0.5)  →  near-boundary path
  └─► Temperature sweep (configured by probe_temperatures, e.g. [0.3, 0.5])
        ├─► All pass?            → THRESHOLD_MARGINAL (score was borderline)
        ├─► First pass, last fail → GENERATOR_TEMPERATURE (lower temp fixes it)
        ├─► Mixed results?       → THRESHOLD_MARGINAL
        └─► All fail?            → proceed to prompt variants (strict_grounding, ...)

Score < probe_score_split  →  clearly-failing path
  └─► Strict grounding prompt first
        ├─► Pass? → recovered (FailureMode: GENERATOR_PARAMETRIC)
        └─► Fail? → temperature sweep → remaining prompt variants
```

**Call budget:**

| Outcome | LLM calls consumed |
|---------|------------------|
| `rejected_above_threshold` (DPO contrast failure) | 0 — early exit |
| Temperature sweep resolves at T=0.3 | 1 |
| Temperature sweep resolves at T=0.5 | 2 |
| Strict grounding resolves immediately | 1 |
| All probes exhausted | 5 (worst case) |

---

## Failure mode taxonomy

| Mode | Gate | What it means |
|------|------|--------------|
| `GENERATOR_TEMPERATURE` | Hallucination | High temperature caused drift from source; lower temp fixes it |
| `GENERATOR_PARAMETRIC` | Hallucination | Model answered from prior knowledge, ignored source passage |
| `SOURCE_AMBIGUOUS` | Hallucination | Source/response relationship unclear to the judge |
| `THRESHOLD_MARGINAL` | Hallucination | Score just below threshold; borderline case |
| `INSTRUCTION_QUALITY` | Reward | Generated question is poorly formed |
| `RESPONSE_QUALITY` | Reward | Answer is on-topic but too shallow; or DPO contrast failure (0 calls) |
| `DOMAIN_MISMATCH` | Reward | Prompt framing wrong for the domain |
| `NEAR_DUPLICATE` | Diversity | Too similar to an accepted sample; not recoverable |
| `UNKNOWN` | Any | All probes failed; inconclusive |

---

## Probe configuration

```python
CuratorConfig(
    enable_diagnostic_probe = True,
    probe_temperatures      = [0.3, 0.5],     # temperature sweep values
    probe_score_split       = 0.5,            # routing boundary
    probe_generator_model   = None,           # None = use llm_model
    probe_extra_templates   = {},             # override named prompt templates (see below)
)
```

### Custom probe templates

The probe selects from named templates, and its routing is fixed to named paths — `probe_extra_templates` does not add new probe paths. What it does:

- **Overriding `strict_grounding`, `domain_specific`, or `default`** replaces the prompt the probe uses on that path. This is the main use case.
- **Overriding `generate_question` has no effect** — the question-regeneration path always uses the built-in template.
- **New keys** are stored but only used when a sample's `metadata["domain_prompt_key"]` names one of them: the domain-variant probe then uses that template instead of `domain_specific`. Keys that no sample's metadata points to are never selected.

```python
CuratorConfig(
    enable_diagnostic_probe = True,
    probe_extra_templates   = {
        "strict_grounding": (
            "Answer this question using ONLY the provided passage. "
            "Quote specific sentences where relevant.\n\n"
            "Passage:\n{source}\n\nQuestion:\n{question}"
        ),
        "domain_specific": (
            "You are a legal analyst. Answer using only the passage text. "
            "Be precise about dates, parties, and obligations.\n\n"
            "Passage:\n{source}\n\nQuestion:\n{question}"
        ),
    },
)
```

**Built-in template keys:**

| Key | Used when | Overridable via `probe_extra_templates` |
|-----|----------|------------------------------------------|
| `strict_grounding` | GENERATOR_PARAMETRIC path — force passage-only answer | Yes |
| `domain_specific` | DOMAIN_MISMATCH path — domain-adapted prompt (unless `metadata["domain_prompt_key"]` selects another key) | Yes |
| `generate_question` | INSTRUCTION_QUALITY path — regenerate the question | No — built-in always used |
| `default` | Temperature-sweep regenerations | Yes |

Template variables: `{source}` (source text) and `{question}` (the instruction/question).

---

## Reward refiner

Runs post-pipeline on samples that RewardGate still rejected after the inline probe. It reads the weakest scoring dimension from the gate's provenance, rewrites the answer targeting that specific axis, then re-evaluates.

```python
CuratorConfig(
    enable_reward_refiner            = True,
    reward_refine_prompt_template    = None,   # None = built-in template
    reward_instruction_refine_template = None, # None = built-in; for question rewrites
)
```

**What it skips:** Samples with `rejected_above_threshold` (DPO contrast failures). These cannot be fixed by rewriting — the generation contrast needs to be fixed at the source.

**Refiner output metadata:**
```json
{
  "reward_refined": true,
  "refinement_type": "answer_rewrite",
  "refinement_axis": "depth"
}
```

For DPO pairs: the refined answer becomes the new `chosen`; the original adversarial response remains as `rejected`.

---

## Reading diagnostics

When `enable_diagnostic_probe=True`, `result.diagnostics` is populated:

```python
d = result.diagnostics.to_dict()

d["total_diagnosed"]    # how many rejected samples were diagnosed
d["probe_recovered"]    # how many were recovered by the probe
d["probe_recovery_pct"] # recovery rate as a fraction (0.0-1.0)
d["total_probe_calls"]  # total LLM calls consumed by the probe
d["mode_counts"]        # {"generator_temperature": 14, "response_quality": 9, ...}
```

**Interpreting `mode_counts`:**
- High `generator_temperature` → lower `llm_temperature` or try `probe_temperatures=[0.2, 0.4]`
- High `generator_parametric` → add a `strict_grounding` template override
- High `response_quality` with 0 probe calls → `rejected_above_threshold`; fix in generation config
- High `instruction_quality` → generated questions are weak; adjust the generation task's `prompt_template` or `difficulty`

---

## When NOT to enable recovery

**`adversarial_qa`**: The HallucinationGate is the intended filter — injected samples should fail. Enabling the probe will attempt to repair them back to grounded answers, defeating the purpose of adversarial generation.

**`rejected_above_threshold`**: The probe always exits immediately for these (0 LLM calls, no recovery). Save the budget by inspecting `mode_counts` first — if all failures are `response_quality`, the probe is not providing value and the generation config needs fixing instead.

---

## Next: [Exporters →](exporters.md)

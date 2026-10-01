# Synthetic Generation

CuratorKIT can generate synthetic training data from your source documents or existing datasets using eight generation tasks. All tasks are **corpus-aware** — when the input is a raw text chunk (e.g. from a PDF), the source text is embedded in the prompt and also stored in `sample.input` so downstream quality gates can verify grounding.

Set `generation_task` in `CuratorConfig` to enable a task. A `llm_model` is required.

---

## Corpus-awareness

When a source sample has `task_type="language_modeling"` (e.g. chunks from a PDF), every generation task extracts the source text and:
- Injects it into the prompt so the LLM answers from the passage
- Sets `output_sample.input = source_text` so HallucinationGate can verify the answer against the exact source

When the input already has an `instruction` field (e.g. a cleaned instruction-following dataset), the task uses that instruction directly and skips the source injection.

---

## Task reference

---

### `qa` — Question-Answer pairs

Generates `num_questions` question-answer pairs per source chunk. Each pair becomes one `DataSample`.

**Prompt structure:**
```
Given the following passage, generate {num_questions} {difficulty} questions and answers.
Each question must be answerable strictly from the passage.

Passage:
---
{source_text}
---

Return JSON: [{"question": "...", "answer": "..."}, ...]
```

**Output fields:** `instruction=question`, `input=source_text`, `output=answer`

```python
CuratorConfig(
    dataset          = "docs/handbook.pdf",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "qa",
    num_questions    = 3,
    difficulty       = "medium",    # "easy" | "medium" | "hard"
)
```

---

### `preference` — DPO preference pairs

Generates a `(chosen, rejected)` pair per source chunk. The rejected response uses a specific degradation pattern — not a hallucination — so both responses are factually correct but differ in depth.

**Prompt structure (single_call, corpus mode):**
```
Generate:
1. A question answerable from this passage
2. A HIGH-QUALITY chosen response — thorough, cites specific details
3. A LOWER-QUALITY rejected response using ONE degradation pattern:
   - Omit the most important specific detail the passage provides
   - Use vague language where the passage is concrete
   - Miss a key distinction the passage explicitly makes

Source passage:
---
{source_text}
---

Return JSON: {"question": "...", "chosen": "...", "rejected": "...", "degradation_pattern": "..."}
```

**Output fields:** `instruction=question`, `input=source_text`, `chosen=chosen`, `rejected=rejected`

**Two generation modes:**
- `single_call` (default) — one LLM call generates question + chosen + rejected together
- `two_pass` — separate LLM calls for chosen and rejected; rejected generated at `temperature + 0.3`

```python
CuratorConfig(
    dataset          = "docs/handbook.pdf",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "preference",
    preference_mode  = "single_call",   # or "two_pass"
)
```

**Note on RewardGate with preference pairs:** The gate dual-scores — `chosen_score >= threshold` AND `rejected_score < threshold`. If `rejected_score >= threshold` (rejected response too good), the pair fails with `rejected_above_threshold`. This is a generation contrast problem, not a gate issue. See [Quality filtering](quality-gates.md).

---

### `grpo` — Group rollouts for GRPO training

Generates `num_responses` candidate responses per prompt. Optionally scores each response using an LLM judge. Produces one output sample with `responses=[r1, r2, ...]` and `reward_scores=[s1, s2, ...]`.

**Temperature control (priority order):**
1. `grpo_temperatures` — explicit list, one temperature per rollout; cycled if shorter than `num_responses`
2. `grpo_temperature_spread` — evenly-spaced range around `llm_temperature`; default `0.0` (all same)
3. Fallback — all rollouts at `llm_temperature`

```python
CuratorConfig(
    dataset              = "data/prompts.jsonl",
    llm_model            = "openai/gpt-4o-mini",
    generation_task      = "grpo",
    num_responses        = 6,
    grpo_temperatures    = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5],  # one per rollout
    score_responses      = True,
    grpo_scoring_llm_model = None,   # None = use llm_model for scoring too
)
```

**Output fields:** `instruction=prompt`, `input=source_text`, `responses=[...]`, `reward_scores=[...]`

---

### `multiturn` — Multi-turn conversations

Generates a full conversation. Default mode (`turn_by_turn`) makes each turn a separate LLM call conditioned on all real prior turns — this gives training data that matches how actual RLHF conversations are collected.

**Turn-by-turn prompt structure:**

*Opening question* (if no instruction exists):
```
Based on the source passage, generate one specific question that opens a learning conversation.
Source: {source_text}
```

*Assistant turns:*
```
Answer the user's question based on the source passage.
Source: {source_text}
Conversation so far: {prior_turns}
Answer the latest question (2-5 sentences, grounded in the passage).
```

*User follow-up turns:*
```
Based on the source passage and conversation, write ONE natural follow-up question.
Source: {source_text}
Conversation so far: {prior_turns}
```

**LLM calls:** `2 × num_turns` in turn_by_turn mode (sequential within a sample, concurrent across samples). For a single-call cheaper alternative, construct `MultiTurnTask(mode="single_call")` directly (quality degrades at >4 turns); this mode is not exposed through `CuratorConfig`.

```python
CuratorConfig(
    dataset          = "docs/handbook.pdf",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "multiturn",
    num_turns        = 4,
)
```

**Output fields:** `instruction=first_user_turn`, `input=source_text`, `output=first_assistant_turn`, `metadata['turns']=remaining_turns`

---

### `evol` — Instruction evolution (Evol-Instruct)

Rewrites instructions into more complex variants using one of five strategies. Optionally generates an answer for the evolved instruction in a second LLM call.

**Evolution strategies** (cycled across `num_evolutions` variants per input):

| Strategy | What it does |
|----------|-------------|
| `add_constraints` | Adds 2-3 specific requirements or edge cases |
| `deepen` | Requires deeper domain expertise |
| `concretize` | Replaces generic references with specific examples |
| `increase_reasoning` | Requires multi-step reasoning |
| `broaden` | Expands scope to related sub-topics |

**Prompt structure:**
```
Evolve this instruction using the "{strategy}" strategy.
Original: {instruction}
[Source passage if corpus mode: {source_text}]
Return JSON: {"evolved_instruction": "...", "strategy_applied": "...", "complexity_notes": "..."}
```

```python
CuratorConfig(
    dataset          = "data/instructions.jsonl",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "evol",
    num_evolutions   = 2,         # 2 variants per input, cycling strategies
    generate_answers = True,      # second LLM call to answer the evolved instruction
)
```

**Output fields:** `instruction=evolved_instruction`, `input=source_text`, `output=answer`

---

### `cot` — Chain-of-thought

Two modes:

**`generate` mode** (default) — takes an instruction and generates full `## Reasoning ... ## Answer` output:
```
Solve the following step by step. Show your reasoning before the final answer.
[Source passage if corpus mode: {source_text}]
Instruction: {instruction}
## Reasoning
## Answer
```

**`wrap` mode** — takes an instruction + existing answer and generates the reasoning that leads to it:
```
Given this instruction and its correct answer, generate the reasoning that leads to it.
Instruction: {instruction}
Correct answer: {answer}
Return JSON: {"reasoning": "...", "answer": "..."}
```

Use `wrap` to add CoT to an existing dataset. Use `generate` for synthetic CoT from scratch.

```python
CuratorConfig(
    dataset          = "data/math.jsonl",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "cot",
    cot_mode         = "generate",   # or "wrap"
)
```

**Output fields:** `instruction=instruction`, `input=source_text`, `output="## Reasoning\n...\n## Answer\n..."`

---

### `adversarial_preference` — Rule-based adversarial DPO pairs

Generates faithful QA pairs then injects one adversarial corruption at `injection_rate` probability. The `chosen` response is faithful; the `rejected` response is adversarially corrupted by a specific failure mode.

**Injection types:**

| Type | What it does |
|------|-------------|
| `contradicts_source` | Answer directly contradicts a specific fact in the source |
| `parametric_drift` | Answer uses general world knowledge, ignoring the source entirely |
| `domain_mismatch` | Answer uses vocabulary and framing from a different domain |
| `instruction_quality` | Answer is vague and hedging, avoids directly addressing the question |

```python
CuratorConfig(
    dataset          = "docs/handbook.pdf",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "adversarial_preference",
    injection_rate   = 0.3,
    injection_types  = ["contradicts_source", "parametric_drift"],   # empty list = all types
    injection_seed   = 42,
)
```

**Output fields:** `instruction=question`, `input=source_text`, `chosen=faithful_answer`, `rejected=adversarially_corrupted_answer`

---

### `adversarial_qa` — Multi-strategy adversarial QA

Generates QA pairs where a controlled fraction are produced using one of five adversarial injection strategies. The HallucinationGate then separates grounded from hallucinated answers.

**Injection types:**

| Type | What it does | Expected diagnosis |
|------|-------------|-------------------|
| `contradicts_source` | Answer directly contradicts a specific fact from the source | `GENERATOR_PARAMETRIC` |
| `parametric_drift` | Answer uses general world knowledge, ignoring the source | `GENERATOR_PARAMETRIC` |
| `high_temperature_drift` | Faithful prompt generated at T=1.4 instead of T=0.7 | `GENERATOR_TEMPERATURE` |
| `domain_mismatch` | Answer uses wrong-domain terminology and framing | `DOMAIN_MISMATCH` |
| `instruction_quality` | Question is deliberately vague; answer responds to vague question | `INSTRUCTION_QUALITY` |

```python
CuratorConfig(
    dataset          = "docs/handbook.pdf",
    llm_model        = "openai/gpt-4o-mini",
    generation_task  = "adversarial_qa",
    injection_rate   = 0.4,
    injection_types  = [],   # empty = all five types; or pick specific ones
    num_questions    = 3,
    difficulty       = "medium",
)
```

**Output fields:** `instruction=question`, `input=source_text`, `output=answer`, `metadata['injected_failure']=True/False`, `metadata['injection_type']=str`

---

## LLM configuration

```python
CuratorConfig(
    llm_model        = "openai/gpt-4o-mini",   # any LiteLLM model string
    llm_temperature  = 0.7,
    llm_max_tokens   = 1024,
    llm_concurrency  = 10,                     # concurrent LLM calls
    llm_api_base     = "http://localhost:8000/v1",  # vLLM, Ollama, custom endpoints
    llm_api_key      = "sk-...",               # or set via env var
    llm_extra_body   = {"chat_template_kwargs": {"enable_thinking": False}},
)
```

For a separate judge model (recommended for gates):
```python
CuratorConfig(
    judge_llm_model      = "openai/gpt-4o",    # stronger/different model for judging
    judge_llm_api_base   = None,               # separate endpoint if needed
    judge_llm_temperature = 0.1,               # low temp for deterministic judgements
)
```

---

## Next: [Quality filtering →](quality-gates.md)

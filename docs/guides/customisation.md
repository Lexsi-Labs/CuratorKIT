# Customisation

All CuratorKIT extension points are accessible through `CuratorConfig` — no subclassing required for prompt customisation, custom LLM backends, or custom rubrics. This page covers all the places you can plug in your own logic.

---

## Custom prompt templates

Every generation task accepts a `*_prompt_template` override.

### QA generation

Required variables: `{context}`, `{num_questions}`. Optional: `{difficulty}`.

Note: if your template omits `{difficulty}` and `difficulty` is set to anything other than `"medium"`, a `Difficulty level: ...` line is appended after the template is rendered. Include the `{difficulty}` placeholder to control its position yourself.

```python
CuratorConfig(
    generation_task   = "qa",
    qa_prompt_template = (
        "You are a legal analyst. Generate {num_questions} precise questions "
        "from the clause below. Each question must be answerable from the text alone.\n\n"
        "Clause:\n{context}\n\n"
        "Return JSON: [{{\"question\": \"...\", \"answer\": \"...\"}}]"
    ),
)
```

### Preference pairs

Required variables: `{instruction}`, `{context_section}`

```python
CuratorConfig(
    generation_task            = "preference",
    preference_prompt_template = (
        "Generate a chosen/rejected pair for DPO training.\n\n"
        "{context_section}"
        "Instruction: {instruction}\n\n"
        "chosen = comprehensive answer with all relevant details\n"
        "rejected = correct but missing the single most critical detail\n\n"
        "Return JSON: {{\"chosen\": \"...\", \"rejected\": \"...\", \"degradation_pattern\": \"...\"}}"
    ),
)
```

### GRPO response prompt

Required variable: `{instruction}`

```python
CuratorConfig(
    generation_task    = "grpo",
    grpo_prompt_template = "Answer the following concisely in under 100 words.\n\n{instruction}",
)
```

### CoT prompt

Required variables for `generate` mode: `{instruction}`
Required variables for `wrap` mode: `{instruction}`, `{answer}`

```python
CuratorConfig(
    generation_task  = "cot",
    cot_mode         = "generate",
    cot_prompt_template = (
        "Solve step by step. Show your work.\n\n"
        "Problem: {instruction}\n\n"
        "## Steps\n(numbered reasoning)\n\n## Answer\n(final answer)"
    ),
)
```

### Evol-Instruct prompt

Required variables: `{instruction}`, `{strategy}`, `{context}`

```python
CuratorConfig(
    generation_task      = "evol",
    evol_prompt_template = (
        "Make this instruction harder using the '{strategy}' strategy.\n\n"
        "Original: {instruction}\nContext: {context}\n\n"
        "Return JSON: {{\"evolved_instruction\": \"...\", \"strategy_applied\": \"{strategy}\", \"complexity_notes\": \"...\"}}"
    ),
)
```

### Multi-turn (single_call mode only)

Required variables: `{num_turns}`, `{context_section}`, `{initial_question}`

The template is only used in `single_call` mode, and `CuratorConfig` always builds
the multi-turn task in `turn_by_turn` mode — so set a custom template by
constructing the task directly:

```python
from curatorkit.generators.multiturn_gen import MultiTurnTask

task = MultiTurnTask(
    llm  = my_llm,
    mode = "single_call",
    prompt_template = (
        "Generate a {num_turns}-turn Q&A conversation.\n\n"
        "{context_section}\n"
        "Opening question: \"{initial_question}\"\n\n"
        "Return JSON: {{\"turns\": [{{\"role\": \"user\", \"content\": \"...\"}}, ...]}}"
    ),
)
```

---

## Custom LLM backends

### Any OpenAI-compatible endpoint (vLLM, Ollama, custom servers)

```python
CuratorConfig(
    llm_model    = "openai/Qwen/Qwen3-8B",
    llm_api_base = "http://localhost:8000/v1",
    llm_api_key  = "token-abc123",         # or set OPENAI_API_KEY env var
)
```

### Ollama (local)

```python
CuratorConfig(
    llm_model    = "ollama/llama3.1:8b",   # "ollama/" prefix routes to OllamaBackend
    llm_api_base = "http://localhost:11434",
)
```

### Model-specific parameters via `llm_extra_body`

Pass any model-specific API parameters that LiteLLM forwards verbatim:

```python
CuratorConfig(
    llm_extra_body = {
        "chat_template_kwargs": {"enable_thinking": True},   # Qwen3 thinking tokens
    },
    # Judge gets thinking disabled — structured output without preamble
    judge_llm_extra_body = {
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

### Separate generator and judge models

Using the same model as both generator and judge causes self-leniency bias — the judge scores its own outputs too generously. Always configure a separate judge model when possible:

```python
CuratorConfig(
    llm_model            = "openai/Qwen/Qwen3-8B",   # generator
    llm_api_base         = "http://localhost:8000/v1",

    judge_llm_model      = "openai/gpt-4o-mini",      # judge — different model
    judge_llm_api_base   = None,                       # None = use standard API
    judge_llm_temperature = 0.1,                       # low temp for deterministic scoring
)
```

---

## Custom reward rubric

The 7 built-in dimensions cover most cases. When they don't fit your domain, replace the entire judge prompt with `reward_prompt_template`. The template must produce a JSON object with `"score"` (float, 0–1) and `"reasoning"` (string). Custom dimension validation is bypassed.

Required variables: `{instruction}`, `{response}`

```python
CuratorConfig(
    reward_threshold        = 0.7,
    reward_prompt_template  = (
        "Rate the following legal answer on a scale of 0.0 to 1.0.\n\n"
        "Criteria:\n"
        "- Cites the specific clause or article (0.4 weight)\n"
        "- States the obligation or right precisely (0.3 weight)\n"
        "- Identifies relevant exceptions or qualifications (0.3 weight)\n\n"
        "Instruction: {instruction}\n\n"
        "Response: {response}\n\n"
        "Respond with JSON only: {{\"score\": 0.XX, \"reasoning\": \"...\"}}"
    ),
)
```

---

## Custom probe templates

`probe_extra_templates` is merged over the built-in probe templates, with your values taking precedence. There are four built-in keys, each tied to a probe path:

| Key | Probe path | Overridable? |
|-----|-----------|--------------|
| `default` | Temperature-sweep re-generation | Yes |
| `strict_grounding` | Strict-grounding probe | Yes |
| `domain_specific` | Domain-grounding probe | Yes |
| `generate_question` | Instruction re-generation probe | No — this probe always uses the built-in template; overriding the key has no effect |

```python
CuratorConfig(
    enable_diagnostic_probe = True,
    probe_extra_templates   = {
        # Override strict_grounding with a domain-specific instruction
        "strict_grounding": (
            "You are a financial analyst. Answer using ONLY the passage. "
            "Cite specific numbers and percentages.\n\n"
            "Passage:\n{source}\n\nQuestion:\n{question}"
        ),
        # Override domain_specific for a legal domain
        "domain_specific": (
            "You are a legal analyst. Answer using only the passage text. "
            "Be precise about dates, parties, and obligations.\n\n"
            "Passage:\n{source}\n\nQuestion:\n{question}"
        ),
    },
)
```

You can also add templates under **new** key names. They are never selected by the default routing, but the domain-grounding probe checks each rejected sample's metadata for a `domain_prompt_key` entry and uses the template with that name instead of `domain_specific`:

```python
CuratorConfig(
    enable_diagnostic_probe = True,
    probe_extra_templates   = {"legal_strict": "Answer as a contracts lawyer, using only the passage.\n\nPassage:\n{source}\n\nQuestion:\n{question}"},
)
# A sample with metadata={"domain_prompt_key": "legal_strict"} is probed
# with the "legal_strict" template on the domain-grounding path.
```

If a sample names a key that doesn't exist, the probe falls back to the `default` template.

Template variables: `{source}` and `{question}`.

---

## Custom preprocessing function

`preprocessing_fn` runs on every raw row before it becomes a `DataSample`. Return `None` to drop the row.

```python
def preprocess(row: dict) -> dict | None:
    # Drop rows with very short output
    if len(row.get("response", "")) < 50:
        return None
    # Rename fields for field_mapping
    row["question"] = row.pop("user_query", "")
    row["answer"]   = row.pop("response", "")
    # Normalise whitespace
    row["answer"] = " ".join(row["answer"].split())
    return row

CuratorConfig(
    dataset          = "data/raw.jsonl",
    preprocessing_fn = preprocess,
    # {source_column: datasample_field} — keys are your columns
    field_mapping    = {"question": "instruction", "answer": "output"},
)
```

---

## Building a custom generator

Subclass `BaseGenerationTask` from `curatorkit.generators.base`. Implement two methods:

```python
from curatorkit.generators.base import BaseGenerationTask
from curatorkit.llm.base import BaseLLM, LLMResponse
from curatorkit.schema import DataSample
import uuid

class SummarisationTask(BaseGenerationTask):
    def _build_messages(self, sample: DataSample) -> list[dict]:
        source = self._get_source_context(sample)
        return [{"role": "user", "content": f"Summarise in 3 sentences:\n\n{source}"}]

    def _parse_response(self, sample: DataSample, response: LLMResponse) -> list[DataSample]:
        text = response.text.strip()
        if not text:
            return []
        return [DataSample(
            id=str(uuid.uuid4()),
            source_uri=sample.source_uri,
            instruction="Summarise the following passage.",
            input=self._get_source_context(sample),
            output=text,
            task_type="instruction_following",
            provenance_chain=list(sample.provenance_chain),
        )]
```

Use it directly with `Pipeline` (bypassing `CuratorConfig`):

```python
from curatorkit.pipeline import Pipeline
from curatorkit.llm.litellm import LiteLLMBackend

llm  = LiteLLMBackend(model="openai/gpt-4o-mini")
task = SummarisationTask(llm=llm, concurrency=10)

pipeline = Pipeline([reader, schema_gate, task, alpaca_exporter], output_dir=Path("output/"))
result   = pipeline.run()
```

---

## Building a custom gate

Subclass `BaseGate` from `curatorkit.interfaces`. Implement `run()`:

```python
from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, RejectedSample

class LengthGate(BaseGate):
    def __init__(self, min_output_words: int = 50):
        self.min_output_words = min_output_words

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        passed, rejected = [], []
        for s in samples:
            word_count = len(s.output.split())
            if word_count >= self.min_output_words:
                passed.append(s)
            else:
                rejected.append(RejectedSample(
                    **s.model_dump(),
                    rejection_reason=f"output_too_short:{word_count}_words",
                    rejecting_step=type(self).__name__,
                ))
        return passed, rejected
```

Insert it directly into a `Pipeline` step list alongside built-in gates.

---

## Next steps

- [ARCHITECTURE.md](../reference/architecture.md) — full contributor-grade reference for all ABCs and contracts
- [Getting started](../getting-started/index.md) — back to the basics

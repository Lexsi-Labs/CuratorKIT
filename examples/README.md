# Examples

Script versions of the main workflows. Each file's docstring states what it
needs and how to run it. The [notebooks](../notebooks/) cover the same ground
interactively with more explanation.

| Example | Needs | What it shows |
|---|---|---|
| [`quickstart/`](quickstart/) | core install only | Declarative YAML pipeline via `curatorkit run`; deliberately messy seed data so every stage visibly fires |
| [`clean_and_dedup.py`](clean_and_dedup.py) | `[hf]` | Clean + MinHash-dedup a Hub dataset, export Alpaca + ShareGPT |
| [`multi_source_ingest.py`](multi_source_ingest.py) | `[hf]` | Merge two Hub datasets and a local file with per-source caps and field mapping |
| [`generate_sft_from_pdf.py`](generate_sft_from_pdf.py) | `[generation,pdf]` + LLM | Gated QA generation from a PDF |
| [`generate_dpo_pairs.py`](generate_dpo_pairs.py) | `[generation,pdf]` + LLM | DPO preference pairs with quality-contrast gating |
| [`adaptive_recovery.py`](adaptive_recovery.py) | `[generation,pdf]` + LLM | Diagnostic probe + reward refiner recovering rejected samples |
| [`hygiene_pipeline.py`](hygiene_pipeline.py) | `[hygiene]` | Secrets, PII, and toxicity stages over contaminated data |

LLM scripts read `CK_MODEL`, `CK_API_BASE`, and the provider's standard key
variable (`OPENAI_API_KEY`); they default to `openai/gpt-4o-mini` and work with
any LiteLLM backend or a local vLLM/Ollama server.

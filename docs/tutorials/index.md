# Tutorials

Hands-on notebooks covering every part of CuratorKIT, from ingestion and cleaning, through LLM generation and adaptive recovery, to data hygiene and a full fine-tuning case study. Each notebook runs standalone: open it on GitHub, or launch it directly in Google Colab.

| # | Tutorial | What you'll learn | Links |
|---|----------|-------------------|-------|
| 01 | **Generate an SFT dataset from a PDF** | Turn any PDF into instruction-following (Alpaca-format) training data, as QA, evolved-instruction, or chain-of-thought tasks, with hallucination and reward gating built in. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/01_generate_sft_dataset.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/01_generate_sft_dataset.ipynb) |
| 02 | **Generate DPO preference pairs** | Build chosen/rejected preference pairs from a PDF, with dual-scored gating that enforces quality contrast between the two answers. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/02_generate_dpo_pairs.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/02_generate_dpo_pairs.ipynb) |
| 03 | **Generate GRPO rollouts** | Two-stage pipeline: turn a PDF into question prompts, then generate multiple reward-scored rollouts per question for GRPO training. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/03_generate_grpo_rollouts.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/03_generate_grpo_rollouts.ipynb) |
| 04 | **Ingest multiple sources** | Merge three heterogeneous datasets (Alpaca, hh-rlhf, GSM8K) with per-source preprocessing functions, sample caps, deduplication, and stratified resampling. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/04_ingest_multiple_sources.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/04_ingest_multiple_sources.ipynb) |
| 05 | **Clean and deduplicate a dataset** | The simplest pipeline: read one dataset with format auto-detection, then deduplicate, clean, filter, and export. No LLM required. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/05_clean_and_dedup.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/05_clean_and_dedup.ipynb) |
| 06 | **Adaptive recovery** | Recover gate-rejected samples instead of discarding them, using inline diagnostic probes and a post-pipeline reward refiner. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/06_adaptive_recovery.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/06_adaptive_recovery.ipynb) |
| 07 | **Adversarial generation** | Use custom prompt templates to generate deliberately contaminated data (credentials, PII, toxic content) for stress-testing the hygiene gates. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/07_adversarial_generation.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/07_adversarial_generation.ipynb) |
| 08 | **Data hygiene pipeline** | Run `SecretsGate`, `PIIPseudonymizer`, and `ToxicityGate` over a contaminated dataset to catch secrets, pseudonymise PII, and reject toxic content with no LLM calls. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/08_data_hygiene_pipeline.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/08_data_hygiene_pipeline.ipynb) |
| 09 | **Case study: filtered vs. unfiltered fine-tuning** | Fine-tune Qwen2.5-1.5B on gate-filtered vs. unfiltered synthetic CUAD data and compare ROUGE-L, BERTScore, and faithfulness. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/09_case_study_filtered_vs_unfiltered.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexsi-Labs/CuratorKIT/blob/main/notebooks/09_case_study_filtered_vs_unfiltered.ipynb) |

## Prefer plain scripts?

Script versions of these workflows live in
[`examples/`](https://github.com/Lexsi-Labs/CuratorKIT/tree/main/examples) — one
file per workflow, with the required extras in each docstring.

## What you'll need

- **No LLM required:** notebooks 04, 05, and 08 run entirely locally and are the best place to start.
- **LLM endpoint required:** notebooks 01-03, 06, 07, and 09 need an OpenAI-compatible endpoint (a local vLLM or Ollama server, or any hosted API). Each notebook includes backend setup instructions.
- **GPU required:** notebook 09 additionally needs a CUDA GPU for the QLoRA fine-tuning steps. A free Colab T4 is enough.

Suggested learning path: start with **05** (cleaning and deduplication), move to **04** (multi-source ingestion), then work through generation (**01-03**), recovery (**06**), and hygiene (**07-08**), and finish with the end-to-end case study (**09**).

# Tutorials

Hands-on notebooks covering every part of CuratorKIT, from ingestion and cleaning, through LLM generation and adaptive recovery, to data hygiene. Each notebook runs standalone: open it on GitHub, or launch it directly in Google Colab.

| # | Tutorial | What you'll learn | Links |
|---|----------|-------------------|-------|
| 01 | **Generate an SFT dataset from a PDF** | Turn any PDF into instruction-following (Alpaca-format) training data, as QA, evolved-instruction, or chain-of-thought tasks, with hallucination and reward gating built in. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/01_generate_sft_dataset.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GqY-OoCz9WdyyUD6Qt9bCFI84bAFb52T?usp=sharing) |
| 02 | **Generate DPO preference pairs** | Build chosen/rejected preference pairs from a PDF, with dual-scored gating that enforces quality contrast between the two answers. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/02_generate_dpo_pairs.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1stGK2iPHUn_MM8xPtA9IlMifENSotXwH?usp=sharing) |
| 03 | **Generate GRPO rollouts** | Two-stage pipeline: turn a PDF into question prompts, then generate multiple reward-scored rollouts per question for GRPO training. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/03_generate_grpo_rollouts.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1U0XcubWNl7397PYx3cFpGxd7NIg2Kioz?usp=sharing) |
| 04 | **Ingest multiple sources** | Merge three heterogeneous datasets (Alpaca, hh-rlhf, GSM8K) with per-source preprocessing functions, sample caps, deduplication, and stratified resampling. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/04_ingest_multiple_sources.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SPG3mGHRVME1TrXM3Y1As6y_Tj3VjQSX?usp=sharing) |
| 05 | **Clean and deduplicate a dataset** | The simplest pipeline: read one dataset with format auto-detection, then deduplicate, clean, filter, and export. No LLM required. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/05_clean_and_dedup.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JmF5MaE5cutvTB3LkwGlrr5nBc_c1dc3?usp=sharing) |
| 06 | **Adaptive recovery** | Recover gate-rejected samples instead of discarding them, using inline diagnostic probes and a post-pipeline reward refiner. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/06_adaptive_recovery.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OYGOUUWqH_HzgimTHq9R11eg6pyNjWs_?usp=sharing) |
| 07 | **Adversarial generation** | Use custom prompt templates to generate deliberately contaminated data (credentials, PII, toxic content) for stress-testing the hygiene gates. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/07_adversarial_generation.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DSWNeHPI3elIL9Ts9U8s7DfFffTrPhoo?usp=sharing) |
| 08 | **Data hygiene pipeline** | Run `SecretsGate`, `PIIPseudonymizer`, and `ToxicityGate` over a contaminated dataset to catch secrets, pseudonymise PII, and reject toxic content with no LLM calls. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/08_data_hygiene_pipeline.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HSXAKGSdXTPSw4CNN269Qj5N6ca0H89R?usp=sharing) |
| 09 | **Filtered vs unfiltered fine-tuning** | Run `HallucinationGate` and `RewardGate` over a 4,500-sample synthetic QA corpus, sample disjoint pools of passed and rejected rows, then fine-tune the same base model (Qwen3-1.7B) twice and compare ROUGE-L, BERTScore-F1, and Faithfulness to quantify the impact of curation. | [GitHub](https://github.com/Lexsi-Labs/CuratorKIT/blob/main/notebooks/09_corpus_filtered_vs_unfiltered_ft.ipynb) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sJ_LL-f4VbVGMol2F-Y3XppIvnU1JXUB?usp=sharing) |

## Prefer plain scripts?

Script versions of these workflows live in
[`examples/`](https://github.com/Lexsi-Labs/CuratorKIT/tree/main/examples) — one
file per workflow, with the required extras in each docstring.

## What you'll need

- **No LLM required:** notebooks 04, 05, and 08 run entirely locally and are the best place to start.
- **LLM endpoint required:** notebooks 01-03, 06, and 07 need an OpenAI-compatible endpoint (a local vLLM or Ollama server, or any hosted API). Each notebook includes backend setup instructions.
- **LLM endpoint + GPU required:** notebook 09 requires an LLM judge endpoint and a GPU (≥16 GB VRAM) for the fine-tuning stage.

Suggested learning path: start with **05** (cleaning and deduplication), move to **04** (multi-source ingestion), then work through generation (**01-03**), recovery (**06**), and hygiene (**07-08**).

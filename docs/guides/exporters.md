# Exporters

Exporters write accepted samples to disk. Multiple formats can be written in a single run. Specify them in `export_formats`:

```python
CuratorConfig(
    export_formats = ["alpaca", "sharegpt", "dpo"],
    output_dir     = "output/",
)
```

---

## Exporter reference

| Format key | Output file | JSON structure |
|------------|------------|----------------|
| `alpaca` | `sft_alpaca.jsonl` | `{"instruction": "...", "input": "...", "output": "..."}` |
| `sharegpt` | `sft_sharegpt.jsonl` | `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}` |
| `dpo` | `dpo.jsonl` | `{"prompt": "...", "chosen": "...", "rejected": "..."}` |
| `grpo` | `grpo.jsonl` | `{"prompt": "...", "responses": [...], "rewards": [...]}` |
| `ppo` | `ppo.jsonl` | `{"prompt": "..."}` |
| `corpus` | `corpus.jsonl` | Full source chunk with metadata (page, heading, chunk_index, ...) |

All exporters **overwrite** files of the same name in `output_dir` on every run — they do not append. To keep the results of a previous run, point each run at a fresh `output_dir` (or move the files elsewhere before re-running).

---

## Compatibility matrix

Not all exporters handle all task types. Incompatible samples are silently skipped (e.g. DPOExporter skips samples without `chosen`/`rejected`).

| Task | alpaca | sharegpt | dpo | grpo | ppo | corpus |
|------|:------:|:--------:|:---:|:----:|:---:|:------:|
| `qa` | ✓ | ✓ | — | — | — | — |
| `preference` | ✓ (chosen) | — | ✓ | — | — | — |
| `grpo` | — | — | — | ✓ | ✓ | — |
| `multiturn` | ✓ (1st turn) | ✓ | — | — | — | — |
| `evol` | ✓ | ✓ | — | — | — | — |
| `cot` | ✓ | ✓ | — | — | — | — |
| `adversarial_preference` | — | — | ✓ | — | — | — |
| `adversarial_qa` | ✓ | ✓ | — | — | — | — |
| Source chunks only | — | — | — | — | — | ✓ |

It is safe to include `dpo` in `export_formats` for mixed-task pipelines — it simply won't write rows for non-preference samples.

---

## Format details

### Alpaca (`sft_alpaca.jsonl`)

Standard three-field SFT format compatible with most fine-tuning frameworks.

```json
{"instruction": "What is backpropagation?", "input": "", "output": "Backpropagation is..."}
```

`input` is the source context (populated for corpus-grounded samples). Empty string when not applicable.

### ShareGPT (`sft_sharegpt.jsonl`)

Conversation format. For multi-turn samples, the full turn list is encoded from `metadata['turns']`:

```json
{
  "conversations": [
    {"from": "human", "value": "What does the study reveal about..."},
    {"from": "gpt",   "value": "The study reveals that..."},
    {"from": "human", "value": "Can you elaborate on the methodology?"},
    {"from": "gpt",   "value": "The methodology involved..."}
  ]
}
```

For non-multi-turn samples, produces a two-turn conversation (human + gpt).

### DPO (`dpo.jsonl`)

```json
{"prompt": "What is...", "chosen": "A thorough answer citing...", "rejected": "A vague answer that omits..."}
```

For conversational preference pairs, `prompt` and both responses are encoded as turn lists.

### GRPO (`grpo.jsonl`)

```json
{
  "prompt": "Explain the significance of...",
  "responses": ["Answer A...", "Answer B...", "Answer C..."],
  "rewards": [0.82, 0.61, 0.74]
}
```

`rewards` is populated only when `score_responses=True` (the default).

### PPO (`ppo.jsonl`)

Prompt-only format for PPO rollout collection:

```json
{"prompt": "Explain the significance of..."}
```

### Corpus (`corpus.jsonl`)

Full metadata preservation for source chunks:

```json
{
  "text": "The document establishes four categories...",
  "source_uri": "docs/handbook.pdf",
  "page": 3,
  "heading": "Section 2 — Categories",
  "chunk_index": 12,
  "content_type": "text",
  "task_type": "language_modeling"
}
```

---

## Train / val / test splits

Set `output_split` to split accepted samples across separate subdirectories. Fractions must sum to 1.0.

```python
CuratorConfig(
    export_formats = ["alpaca", "dpo"],
    output_split   = {"train": 0.8, "val": 0.1, "test": 0.1},
    output_dir     = "output/",
)
```

Output structure:
```
output/
  train/
    sft_alpaca.jsonl
    dpo.jsonl
  val/
    sft_alpaca.jsonl
    dpo.jsonl
  test/
    sft_alpaca.jsonl
    dpo.jsonl
```

Samples are shuffled before splitting. The last split receives any remainder from rounding.

---

## Output directory layout

Every run writes these files regardless of which exporters are configured:

```
output/
  manifest.json             Pipeline config hash, stage counts, rejection breakdown
  rejected.jsonl            All rejected samples with structured reasons
  dataset_card.md           Human-readable run summary
  checksums.txt             SHA-256 for all output files
  diagnostic_summary.json   Failure mode counts, recovery stats (when probe active)
  [format files]            Only the formats you listed in export_formats
```

`manifest.json` and `rejected.jsonl` are always written and cannot be disabled. They are the primary audit trail for the run.

---

## Next: [Customisation →](customisation.md)

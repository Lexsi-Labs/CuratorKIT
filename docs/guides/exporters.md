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

Every exporter enforces this table as a hard per-sample gate: a sample whose task_type isn't listed as compatible with that format is **skipped and counted in a warning**, even if its fields happen to be non-empty (e.g. `instruction_following` data is not silently accepted by GRPO/PPO/Corpus just because `instruction`/`output` are populated — only their own designed task_type is). Nothing crashes and no file is ever refused to be written — an incompatible combination just produces an empty (or emptier) file plus a warning explaining why, rather than silently writing lossy or meaningless rows.

- **DPO / GRPO / PPO / Corpus** skip a sample when its task_type isn't in the accepted set below, *or* when a field it still needs is empty even for an accepted task_type (`chosen`/`rejected`; `instruction`; `instruction`; `output`-or-`input`, respectively) — one summary warning with the count and the task_type(s) seen.
- **GRPO** additionally never warns on empty `responses`/`rewards` alone within task_type `"grpo"` — that's the documented, intentional case of exporting seed prompts before `GRPORolloutTask` has run.
- **Alpaca / ShareGPT** also gate on task_type now (so, notably, a `conversational` sample is no longer silently truncated to its first turn by Alpaca — it's skipped, with a warning pointing at `"sharegpt"` instead). Within an accepted task_type, a genuinely empty `instruction`/`output` still gets written (it's a real, if broken, sample) but warns separately.

| task_type | alpaca | sharegpt | dpo | grpo | ppo | corpus |
|------|:------:|:--------:|:---:|:----:|:---:|:------:|
| `instruction_following` (qa, evol, cot, adversarial_qa) | ✓ | ✓ | — | — | — | — |
| `conversational` (multiturn) | — | ✓ | — | — | — | — |
| `preference`, `implicit_preference` (preference, adversarial_preference) | — | — | ✓ | — | — | — |
| `unpaired_preference` (connector) | ✓ | ✓ | — | — | — | — |
| `grpo` (grpo task) | — | — | — | ✓ | — | — |
| `prompt_only` (connector) | — | — | — | — | ✓ | — |
| `language_modeling` / `source_chunk` (PDF chunks, pretrain text) | — | — | — | — | — | ✓ |

This table is `curatorkit.exporters.compatibility.TASK_TYPE_EXPORT_FORMATS` — the same table `export_formats=None` (the default) uses to pick formats automatically:

- With `generation_task` set, the task_type it produces is known ahead of time, so the matching format(s) above are selected before the pipeline even runs.
- With no `generation_task` (plain curation/connector ingestion), the task_type is instead identified from what the readers actually emit, so the matching format(s) are picked after reading.

Set `export_formats` explicitly to override — e.g. to add a deliberately partial/lossy export (`ppo` alongside `grpo`'s rollouts, or `corpus` to dump any task's `output` as plain text). Combining an explicit `export_formats` with a `generation_task` it doesn't match (e.g. `["dpo"]` with `generation_task="qa"`) emits a warning rather than being silently corrected or blocked — it's still respected as given.

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

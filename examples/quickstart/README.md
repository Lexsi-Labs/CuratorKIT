# CuratorKIT Quickstart

Produce a training-ready SFT dataset, with full provenance, in under five minutes.

This example ingests a small, deliberately messy seed file and shows every stage of a
CuratorKIT pipeline doing real work: malformed lines are rejected by the reader, invalid
samples are rejected by the schema gate, duplicates are removed, dirty text is cleaned,
and the survivors are exported in two training formats.

## Setup

```bash
pip install curatorkit
git clone https://github.com/Lexsi-Labs/CuratorKIT.git
cd CuratorKIT
```

(Or install straight from source: `pip install "curatorkit[all] @ git+https://github.com/Lexsi-Labs/CuratorKIT.git"`)

## Run the pipeline

From the repo root:

```bash
curatorkit run examples/quickstart/pipeline.yaml --output-dir ./out/
```

## What each stage does

The pipeline in [`pipeline.yaml`](pipeline.yaml) runs four stages in order:

| Stage | Type | What it does here |
|-------|------|-------------------|
| `jsonl` reader | reader | Loads `seed_data.jsonl`, auto-detects the Alpaca-style row format, and rejects any line that is not valid JSON. |
| `schema` gate | gate | Requires non-empty `instruction` and `output`, 10-2048 tokens combined, and clean encoding. Every failure is rejected with a structured reason. |
| `exact_dedup` | normalizer | Drops exact duplicates (case- and whitespace-insensitive hash of instruction + output). |
| `text_cleaner` | normalizer | Strips HTML tags, fixes mojibake (`donât` → `don't`), collapses whitespace, removes control characters. |
| `alpaca` + `sharegpt` | exporters | Write the surviving samples in both formats. |

## The seed data is messy on purpose

`seed_data.jsonl` has 15 lines: 6 clean samples and 9 problem lines, each crafted to
trigger a specific stage (lines 7-8 are dirty but salvageable: the cleaner fixes them
in place, so 8 samples survive to export):

| Line | Problem | Caught by | Result |
|------|---------|-----------|--------|
| 7 | Output wrapped in `<div><p><b>...` HTML | `text_cleaner` | cleaned, kept |
| 8 | Mojibake (`doesnât`) + runs of spaces/tabs | `text_cleaner` | cleaned, kept |
| 9 | Exact duplicate of line 1 | `exact_dedup` | dropped |
| 10 | Duplicate of line 2 up to casing/whitespace | `exact_dedup` | dropped |
| 11 | Empty `output` | `schema` gate | rejected: `missing_field:output` |
| 12 | "Hi" / "Hello!" (only 2 tokens) | `schema` gate | rejected: `below_min_tokens:2` |
| 13 | Whitespace-only `instruction` | `schema` gate | rejected: `missing_field:instruction` |
| 14 | Null byte embedded in `output` | `schema` gate | rejected: `encoding_error:null_byte_in_output` |
| 15 | Truncated line, not valid JSON | `jsonl` reader | rejected: `json_decode_error:...` |

## Expected output

The run exits 0 and writes:

```
out/
  sft_alpaca.jsonl     # 8 samples, Alpaca format (instruction/input/output)
  sft_sharegpt.jsonl   # 8 samples, ShareGPT conversation format
  manifest.json        # Full provenance manifest with per-stage counts
  dataset_card.md      # Human-readable dataset card
  rejected.jsonl       # 5 rejected samples, each with a structured reason
  checksums.txt        # SHA-256 checksums for the output files
```

The stage counts in `manifest.json` (also rendered as a table in `dataset_card.md`):

```
JSONLReader        14 read,  1 rejected   (the truncated JSON line)
SchemaGate         14 in →  10 out,  4 rejected
ExactDeduplicator  10 in →   8 out   (2 exact duplicates removed)
TextCleaner         8 in →   8 out   (2 samples cleaned in place)
AlpacaExporter      8 exported
ShareGPTExporter    8 exported
```

And the rejection breakdown:

```
json_decode_error:Unterminated string starting at: line 1 column 68 (char 67)  1
missing_field:output                                                           1
below_min_tokens:2                                                             1
missing_field:instruction                                                      1
encoding_error:null_byte_in_output                                             1
```

Nothing is dropped silently: every one of the 15 input lines is accounted for as
either an exported sample, a logged duplicate, or an entry in `rejected.jsonl`.

You can see the cleaner's effect in the exported files. The HTML-wrapped dropout
answer comes out as plain prose:

```json
{"instruction": "What is dropout in neural networks?", "input": "", "output": "Dropout is a regularization technique that randomly sets a fraction of activations to zero during training. This prevents units from co-adapting and improves generalization. At inference time, dropout is disabled."}
```

## Dry run (validate config without processing)

```bash
curatorkit run examples/quickstart/pipeline.yaml --dry-run
```

Prints the resolved step plan and exits without reading any data.

## Next steps

- Point `readers[0].path` at your own JSONL file; format detection handles
  Alpaca, ShareGPT, preference pairs, and more
- Adjust `gates[0].min_tokens` / `max_tokens` for your domain
- Add `- type: minhash_dedup` to the normalizers for near-duplicate removal
- Set `use_tiktoken: true` in the gate for exact LLM token counts
  (requires `pip install "curatorkit[tiktoken]"`)
- Explore the [notebooks](../../notebooks/) for LLM generation and data
  hygiene pipelines, and the [docs](https://lexsi-labs.github.io/CuratorKIT/)
  for the full configuration reference

# Data Sources

CuratorKIT reads data through **connectors** — one per source format. All connectors produce `DataSample` objects with a consistent schema regardless of the input format.

---

## Reader reference

| Reader | Triggered by | Notes |
|--------|-------------|-------|
| `JSONLReader` | `.jsonl` extension | One JSON object per line |
| `JSONReader` | `.json` extension | Top-level array or object |
| `CSVReader` | `.csv` / `.tsv` extension | Tab-separated auto-detected |
| `ParquetReader` | `.parquet` extension | Requires `pip install "curatorkit[parquet]"` |
| `HuggingFaceReader` | Any string without a file extension | Requires `pip install "curatorkit[hf]"` |
| `PDFReader` | `.pdf` extension | Requires `pip install "curatorkit[pdf]"` |

---

## Automatic format detection

When you don't specify `field_mapping`, the connector inspects the first few rows and infers the format. Three detection layers run in order:

1. **Column name matching** — looks for `instruction`, `output`, `chosen`, `prompt`, etc.
2. **Value type validation** — confirms matched columns contain strings
3. **Role alias normalisation** — `"human"` / `"user"` → `instruction`; `"assistant"` / `"gpt"` → `output`

This handles Alpaca, ShareGPT, plain-text, and most HuggingFace dataset shapes without configuration.

When detection fails or produces the wrong mapping, override it explicitly:

```python
CuratorConfig(
    dataset       = "data/my_dataset.jsonl",
    field_mapping = {
        "question": "instruction",   # your column → DataSample field
        "answer":   "output",
        "context":  "input",
    },
)
```

> **Mapping direction:** each **key** is a column in *your* data; each **value** is the
> `DataSample` field it maps to — `{source_column: datasample_field}`. Getting this
> backwards fails silently: the keys won't match any source columns, so nothing is renamed.

Keys may use dot notation to reach nested dict values, e.g. `{"meta.prompt": "instruction"}`
pulls `row["meta"]["prompt"]` into `instruction`. Dot notation traverses dicts only — list
indexing (e.g. `messages[0]`) is not supported.

---

## Common field mapping examples

**Plain text corpus (one document per row):**
```python
field_mapping = {"text": "output"}   # maps your "text" column → sample.output
```

**Nested source keys (dot notation):**
```python
field_mapping = {"meta.prompt": "instruction", "meta.response": "output"}
```

**UltraChat / Orca style:**
```python
field_mapping = {"system": "instruction", "question": "input", "response": "output"}
```

**OpenAI chat format:** columns named `messages` (or `conversations`, `chat`, `dialogue`)
holding role/content turn lists are recognised by automatic format detection — no
`field_mapping` needed. If your turn list lives under a non-standard column name, rename it
so detection picks it up:
```python
field_mapping = {"dialog_turns": "messages"}
```
To extract individual turns yourself (field mapping cannot index into lists), use a
[preprocessing function](#preprocessing-function) instead.

---

## Preprocessing function

`preprocessing_fn` runs on the raw row dict before field mapping. Return `None` to drop the row.

```python
def clean_row(row: dict) -> dict | None:
    if len(row.get("answer", "")) < 20:
        return None                          # drop short answers
    row["answer"] = row["answer"].strip()
    return row

CuratorConfig(
    dataset          = "data/raw.jsonl",
    preprocessing_fn = clean_row,
    field_mapping    = {"question": "instruction", "answer": "output"},
)
```

For multiple sources with different preprocessing, pass a list of callables — one per source:

```python
CuratorConfig(
    dataset          = ["data/source_a.jsonl", "data/source_b.jsonl"],
    preprocessing_fn = [preprocess_a, preprocess_b],
)
```

---

## Multi-source pipelines

Pass a list of sources. Each can be a string or a dict with per-source overrides:

```python
CuratorConfig(
    dataset = [
        "tatsu-lab/alpaca",                                    # HF Hub
        "data/extra.jsonl",                                    # local file
        {"name": "openai/summarize_from_feedback",             # with overrides
         "split": "validation",
         "max_samples": 500},
    ],
    split = "train",   # default split for sources that don't override it
)
```

Per-source dict keys: `name`, `split`, `subset`, and `max_samples`. `max_samples` in a per-source dict caps that reader independently before samples are combined. The global `max_samples` in `CuratorConfig` caps the total after all sources are merged.

---

## HuggingFace-specific options

```python
CuratorConfig(
    dataset    = "allenai/dolma",
    split      = "train",
    hf_subset  = "cc_en_head",         # dataset config/subset
    hf_columns = ["text", "id"],       # load only these columns (saves memory on large datasets)
    streaming  = True,                 # streaming mode — no local disk cache
    hf_token   = "hf_...",             # private datasets
)
```

---

## PDF options

PDFReader chunks the document before handing samples to the pipeline. Each chunk becomes one `DataSample` with `task_type="language_modeling"` and `output=chunk_text`.

```python
CuratorConfig(
    dataset              = "docs/report.pdf",
    pdf_chunk_strategy   = "heading",    # "heading" | "sentence" | "fixed"
    pdf_chunk_max_tokens = 512,
    pdf_chunk_overlap_tokens = 50,
    pdf_min_section_tokens   = 30,       # merge sections shorter than this
    pdf_extract_tables       = False,
    pdf_ocr                  = False,    # enable for scanned PDFs
)
```

When `generation_task` is also set, the `pdf_output_mode` field controls whether the PDF reader produces raw chunks (default) or pre-formatted generation inputs:

```python
pdf_output_mode = "chunk"        # raw source chunks → generation task uses them (recommended)
pdf_output_mode = "qa"           # reader generates QA inline (legacy)
pdf_output_mode = "preference"   # reader generates preference pairs inline (legacy)
pdf_output_mode = "grpo"         # reader generates GRPO rollouts inline (legacy)
pdf_output_mode = "multiturn"    # reader generates multi-turn conversations inline (legacy)
```

For new pipelines, use `"chunk"` and set `generation_task` separately — the inline generation modes are legacy and give less control over filtering and recovery.

---

## Token length limits

`SchemaGate` runs immediately after all readers and filters on token length. Tune for your data:

```python
CuratorConfig(
    min_tokens    = 10,      # drop samples shorter than this
    max_tokens    = 4096,    # drop samples longer than this (default 2048)
    use_tiktoken  = False,   # True = tiktoken cl100k; False = whitespace split (fast)
)
```

---

## Next: [Generation →](generation.md)

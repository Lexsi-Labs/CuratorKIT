"""Merge multiple sources with per-source caps and field mapping. No LLM required.

Script version of notebook 04. Combines two Hub datasets and a local file,
caps each reader independently, deduplicates across all of them, and exports.

Requires: pip install "curatorkit[hf]"
Run:      python examples/multi_source_ingest.py
"""

import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig

# A local file as the third source
local = Path("output/local_extra.jsonl")
local.parent.mkdir(parents=True, exist_ok=True)
local.write_text(
    json.dumps(
        {
            "q": "What does a provenance manifest record in a curation pipeline?",
            "a": "Per-stage sample counts, the config hash, and a structured rejection breakdown.",
        }
    )
    + "\n"
)

result = Curator(
    CuratorConfig(
        dataset=[
            {"name": "tatsu-lab/alpaca", "max_samples": 500},
            {"name": "openai/gsm8k", "subset": "main", "max_samples": 200},
            {"name": str(local)},
        ],
        field_mapping={
            "q": "instruction",
            "a": "output",
            "question": "instruction",
            "answer": "output",
        },
        dedup="exact",
        clean=True,
        export_formats=["alpaca"],
        output_dir="output/multi_source",
    )
).run()

result.print_summary()

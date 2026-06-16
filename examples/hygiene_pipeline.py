"""Catch secrets, pseudonymize PII, and reject toxic content. No LLM required.

Script version of notebook 08. Runs the three hygiene stages over a small
deliberately contaminated dataset (all credentials are documented fakes).

Requires: pip install "curatorkit[hygiene]"
          python -m spacy download en_core_web_sm
Run:      python examples/hygiene_pipeline.py
"""

import json
from pathlib import Path

from curatorkit import Curator, CuratorConfig

contaminated = [
    {
        "instruction": "How do I configure the S3 client?",
        "output": "Set aws_access_key_id to AKIAIOSFODNN7EXAMPLE in ~/.aws/credentials.",
    },
    {
        "instruction": "Draft a follow-up email to the candidate.",
        "output": "Dear John Smith, reach me at jane.doe@example.com or 555-867-5309.",
    },
    {
        "instruction": "Reply to this forum post.",
        "output": "You absolute idiot, nobody wants your garbage opinions here.",
    },
    {
        "instruction": "Explain what a context manager does in Python.",
        "output": "A context manager wraps setup and teardown around a block via the with statement.",
    },
]

src = Path("output/contaminated.jsonl")
src.parent.mkdir(parents=True, exist_ok=True)
src.write_text("\n".join(json.dumps(r) for r in contaminated))

result = Curator(
    CuratorConfig(
        dataset=str(src),
        secrets_gate=True,
        pii_pseudonymize=True,
        toxicity_gate=True,
        export_formats=["alpaca"],
        output_dir="output/hygiene",
    )
).run()

result.print_summary()
for r in result.rejected:
    print(f"rejected: {r.rejection_reason}")

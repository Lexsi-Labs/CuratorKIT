"""Generation with the full recovery stack: diagnostic probe + reward refiner.

Script version of notebook 06. Rejected samples are diagnosed against the
failure-mode taxonomy; recoverable ones are repaired instead of discarded.
diagnostic_summary.json reports what was recovered and why.

Requires: pip install "curatorkit[generation,pdf]"
          an API key (OPENAI_API_KEY) or a local server via CK_API_BASE
Run:      python examples/adaptive_recovery.py path/to/document.pdf
"""

import json
import os
import sys

from curatorkit import Curator, CuratorConfig

if len(sys.argv) != 2:
    sys.exit("usage: python examples/adaptive_recovery.py <document.pdf>")

result = Curator(
    CuratorConfig(
        dataset=sys.argv[1],
        llm_model=os.getenv("CK_MODEL", "openai/gpt-4o-mini"),
        judge_llm_model=os.getenv("CK_JUDGE_MODEL"),  # separate judge avoids self-leniency
        llm_api_base=os.getenv("CK_API_BASE"),
        generation_task="qa",
        hallucination_threshold=0.7,
        reward_threshold=0.7,
        enable_diagnostic_probe=True,
        enable_reward_refiner=True,
        export_formats=["alpaca"],
        output_dir="output/recovery",
    )
).run()

result.print_summary()
if result.diagnostics is not None:
    print(json.dumps(result.diagnostics.to_dict(), indent=2))

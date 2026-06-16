"""Generate DPO preference pairs from a PDF, with dual-scored gating.

Script version of notebook 02. Each pair gets a strong chosen answer and a
deliberately weaker rejected answer; gating enforces the quality contrast.

Requires: pip install "curatorkit[generation,pdf]"
          an API key (OPENAI_API_KEY) or a local server via CK_API_BASE
Run:      python examples/generate_dpo_pairs.py path/to/document.pdf
"""

import os
import sys

from curatorkit import Curator, CuratorConfig

if len(sys.argv) != 2:
    sys.exit("usage: python examples/generate_dpo_pairs.py <document.pdf>")

result = Curator(
    CuratorConfig(
        dataset=sys.argv[1],
        llm_model=os.getenv("CK_MODEL", "openai/gpt-4o-mini"),
        llm_api_base=os.getenv("CK_API_BASE"),
        generation_task="preference",
        num_questions=2,
        reward_threshold=0.6,
        export_formats=["dpo"],
        output_dir="output/dpo_pairs",
    )
).run()

result.print_summary()

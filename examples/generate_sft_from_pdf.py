"""Generate gated synthetic QA data from a PDF.

Script version of notebook 01. Chunks a PDF, generates QA pairs, and keeps
only answers grounded in their source chunk.

Requires: pip install "curatorkit[generation,pdf]"
          an API key (OPENAI_API_KEY) or a local server via CK_API_BASE
Run:      python examples/generate_sft_from_pdf.py path/to/document.pdf
"""

import os
import sys

from curatorkit import Curator, CuratorConfig

if len(sys.argv) != 2:
    sys.exit("usage: python examples/generate_sft_from_pdf.py <document.pdf>")

result = Curator(
    CuratorConfig(
        dataset=sys.argv[1],
        llm_model=os.getenv("CK_MODEL", "openai/gpt-4o-mini"),
        llm_api_base=os.getenv("CK_API_BASE"),  # e.g. http://localhost:8000/v1 for vLLM
        generation_task="qa",
        num_questions=3,
        hallucination_threshold=0.7,  # grounding gate
        reward_threshold=0.7,  # LLM-judge gate
        export_formats=["alpaca", "sharegpt"],
        output_dir="output/sft_from_pdf",
    )
).run()

result.print_summary()

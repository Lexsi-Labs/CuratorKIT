"""Clean and deduplicate an existing dataset. No LLM or API key required.

Script version of notebook 05. Reads a HuggingFace Hub dataset (capped),
deduplicates with MinHash, cleans the text, and exports Alpaca + ShareGPT.

Requires: pip install "curatorkit[hf]"
Run:      python examples/clean_and_dedup.py
"""

from curatorkit import Curator, CuratorConfig

result = Curator(
    CuratorConfig(
        dataset={"name": "tatsu-lab/alpaca", "max_samples": 2000},
        dedup="minhash",
        clean=True,
        export_formats=["alpaca", "sharegpt"],
        output_dir="output/clean",
    )
).run()

result.print_summary()
result.sample(n=2)

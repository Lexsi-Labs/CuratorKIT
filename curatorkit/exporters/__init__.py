from curatorkit.exporters.alpaca import AlpacaExporter
from curatorkit.exporters.argilla import ArgillaExporter
from curatorkit.exporters.corpus import CorpusExporter
from curatorkit.exporters.dpo import DPOExporter
from curatorkit.exporters.grpo import GRPOExporter
from curatorkit.exporters.messages import MessagesExporter
from curatorkit.exporters.ppo import PPOExporter
from curatorkit.exporters.sharegpt import ShareGPTExporter

# export_formats name -> exporter. The stem of each exporter's `filename` is the
# dataset config name in the output folder's README.md.
EXPORTERS = {
    "alpaca": AlpacaExporter,
    "sharegpt": ShareGPTExporter,
    "messages": MessagesExporter,
    "dpo": DPOExporter,
    "grpo": GRPOExporter,
    "ppo": PPOExporter,
    "corpus": CorpusExporter,
}

__all__ = [
    "EXPORTERS",
    "AlpacaExporter",
    "ArgillaExporter",
    "CorpusExporter",
    "DPOExporter",
    "GRPOExporter",
    "MessagesExporter",
    "PPOExporter",
    "ShareGPTExporter",
]

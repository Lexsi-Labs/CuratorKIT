"""
Integration test: full pipeline -> Alpaca export -> TRL SFTTrainer.

This integration test is run manually (requires the trl extra); it is not part of the default CI matrix. It runs the full pipeline from a 100-sample
seed JSONL through schema gate, exact dedup, text cleaning, and Alpaca export,
then loads the result in a TRL SFTTrainer and asserts the training step
completes without a format exception.

Format regressions in the exporter surface here immediately — not when a
training engineer runs a job.

Marked as 'integration' and 'slow'. Skipped automatically if TRL is not
installed (CI must install [trl] extras to run this test).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_jsonl(path: Path, n: int = 100) -> None:
    """Write n Alpaca-format records to path."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            record = {
                "instruction": f"Explain the concept of {_TOPICS[i % len(_TOPICS)]} in simple terms.",
                "output": (
                    f"Sure! {_TOPICS[i % len(_TOPICS)].capitalize()} is a fundamental concept. "
                    f"Here is a clear explanation with examples. "
                    f"First, let us define the term. "
                    f"Then we will look at practical applications. "
                    f"Sample number {i + 1} of the dataset."
                ),
            }
            f.write(json.dumps(record) + "\n")


_TOPICS = [
    "recursion",
    "gradient descent",
    "transformers",
    "tokenization",
    "attention mechanism",
    "backpropagation",
    "reinforcement learning",
    "supervised learning",
    "unsupervised learning",
    "transfer learning",
    "regularization",
    "overfitting",
    "underfitting",
    "cross-validation",
    "hyperparameter tuning",
    "batch normalization",
    "dropout",
    "embeddings",
    "word2vec",
    "BERT",
]


# ---------------------------------------------------------------------------
# Pipeline run helper
# ---------------------------------------------------------------------------


def _run_pipeline(seed_path: Path, output_dir: Path) -> None:
    from curatorkit.connectors.jsonl import JSONLReader
    from curatorkit.exporters.alpaca import AlpacaExporter
    from curatorkit.gates.schema import SchemaGate
    from curatorkit.manifest import DatasetCardGenerator, ProvenanceManifest
    from curatorkit.normalizers.clean import TextCleaner
    from curatorkit.normalizers.dedup import ExactDeduplicator
    from curatorkit.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            JSONLReader(seed_path, format="alpaca"),
            SchemaGate(required_fields=["instruction", "output"], min_tokens=10, max_tokens=2048),
            ExactDeduplicator(),
            TextCleaner(),
            AlpacaExporter(),
        ],
        output_dir=output_dir,
    )
    result = pipeline.run()

    # Always write manifest and sidecar — they cannot be disabled
    manifest_builder = ProvenanceManifest(result=result, output_dir=output_dir)
    manifest_builder.write()
    manifest_builder.write_rejected_sidecar()
    manifest_data = manifest_builder.build()
    DatasetCardGenerator().generate(manifest_data, output_dir)

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_produces_alpaca_jsonl():
    """End-to-end: seed JSONL -> pipeline -> sft_alpaca.jsonl exists and is valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        seed_path = Path(tmpdir) / "seed.jsonl"
        _make_seed_jsonl(seed_path, n=100)

        _run_pipeline(seed_path, output_dir)

        alpaca_path = output_dir / "sft_alpaca.jsonl"
        assert alpaca_path.exists(), "sft_alpaca.jsonl must be produced"

        lines = alpaca_path.read_text().strip().splitlines()
        assert len(lines) > 0, "Output must not be empty"

        for line in lines:
            record = json.loads(line)
            assert "instruction" in record, f"Missing 'instruction' key: {record}"
            assert "input" in record, f"Missing 'input' key: {record}"
            assert "output" in record, f"Missing 'output' key: {record}"


@pytest.mark.integration
@pytest.mark.slow
def test_manifest_and_sidecar_always_emitted():
    """manifest.json and rejected.jsonl must always be present, even with zero rejections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        seed_path = Path(tmpdir) / "seed.jsonl"
        _make_seed_jsonl(seed_path, n=20)

        _run_pipeline(seed_path, output_dir)

        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "rejected.jsonl").exists()
        assert (output_dir / "dataset_card.md").exists()

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert "stage_counts" in manifest
        assert "rejected_breakdown" in manifest
        assert "wall_clock_seconds" in manifest


@pytest.mark.integration
@pytest.mark.slow
def test_trl_sfttrainer_accepts_alpaca_output():
    """Load the Alpaca output in TRL SFTTrainer and assert one training step completes.

    Skipped if TRL is not installed. Install with: pip install curatorkit[trl]
    """
    pytest.importorskip("trl", reason="TRL not installed — install curatorkit[trl]")
    pytest.importorskip("transformers", reason="transformers not installed")
    pytest.importorskip("datasets", reason="datasets not installed")
    pytest.importorskip("torch", reason="torch not installed")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        seed_path = Path(tmpdir) / "seed.jsonl"
        _make_seed_jsonl(seed_path, n=100)

        _run_pipeline(seed_path, output_dir)

        alpaca_path = output_dir / "sft_alpaca.jsonl"
        assert alpaca_path.exists()

        # Load into HuggingFace Dataset
        raw_records = [json.loads(line) for line in alpaca_path.read_text().strip().splitlines()]
        assert len(raw_records) > 0

        # Format as chat-style text for SFT
        def format_sample(record: dict) -> dict:
            prompt = record["instruction"]
            if record.get("input"):
                prompt = f"{prompt}\n\n{record['input']}"
            return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{record['output']}"}

        formatted = [format_sample(r) for r in raw_records]
        hf_dataset = Dataset.from_list(formatted)

        # Use a tiny model — gpt2 is the smallest publicly available causal LM
        model_name = "sshleifer/tiny-gpt2"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            pytest.skip(f"Could not load model {model_name}: {e}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        training_args = SFTConfig(
            output_dir=str(Path(tmpdir) / "trainer_output"),
            num_train_epochs=1,
            max_steps=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            report_to="none",
            max_seq_length=128,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )

        # This must complete without a format exception
        trainer.train()

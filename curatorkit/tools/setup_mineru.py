"""
MinerU readiness check for CuratorKIT PDF extraction.

With mineru 3.x, model weights are downloaded automatically on first use.
No manual setup, no config file writes, no OCR patching required.

This module exists solely to:
  1. Verify mineru is importable (fail fast with a clear message if not)
  2. Set MINERU_DEVICE_MODE to 'cuda' if available (user preference)

ensure_mineru_ready() and setup_mineru() are kept as stubs for backwards
compatibility but are now no-ops beyond the checks above.
"""

from __future__ import annotations

import os
from pathlib import Path


def ensure_mineru_ready(source: str = "huggingface") -> None:
    """Ensure MinerU is ready for PDF extraction.

    With mineru 3.x this is a no-op beyond verifying the package is installed
    and setting the device. Model weights are downloaded automatically on first use.
    """
    try:
        import mineru  # noqa: F401
    except ImportError:
        raise ImportError(
            "\nMinerU is not installed. Fix:\n\n"
            '    pip install "curatorkit[pdf]"\n\n'
            "A CUDA GPU is used automatically when available (recommended); "
            "CPU works but is slower.\n"
        )

    # Prefer CUDA if available; do not override if user set the env var already
    if "MINERU_DEVICE_MODE" not in os.environ:
        try:
            import torch

            if torch.cuda.is_available():
                os.environ["MINERU_DEVICE_MODE"] = "cuda"
        except ImportError:
            pass


def setup_mineru(
    models_dir: Path | None = None,
    device: str = "cuda",
    source: str = "huggingface",
    verbose: bool = True,
) -> None:
    """No-op stub — mineru 3.x manages model downloads automatically."""
    if verbose:
        print(
            "MinerU 3.x manages model downloads automatically on first use.\n"
            "No manual setup is needed."
        )


def validate_setup(models_dir: Path | None = None) -> tuple[bool, list[str]]:
    """Check that mineru is installed and importable."""
    try:
        import mineru  # noqa: F401

        return True, []
    except ImportError:
        return False, ["mineru package not installed"]

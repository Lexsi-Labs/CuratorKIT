"""Map notebook ``BACKEND`` to LiteLLM ``(model, api_base, api_key)``.

vLLM install + ``vllm serve`` live here so Colab run cells stay AlignTune-short.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request


def resolve_backend(backend: str, model: str) -> tuple[str, str | None, str | None]:
    """Return LiteLLM args for ``openrouter`` | ``vllm`` | ``ollama``."""
    b = (backend or "").strip().lower()
    if b == "openrouter":
        return f"openrouter/{model}", "https://openrouter.ai/api/v1", os.environ.get("OPENROUTER_API_KEY")
    if b == "vllm":
        _ensure_vllm(model)
        return f"openai/{model}", "http://localhost:8000/v1", os.environ.get("HF_TOKEN") or "EMPTY"
    if b == "ollama":
        return f"ollama/{model}", None, None
    raise RuntimeError("Set BACKEND to openrouter, vllm, or ollama")


def _vllm_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=1)
        return True
    except Exception:
        return False


def _ensure_vllm(model: str) -> None:
    if _vllm_up():
        return
    if shutil.which("vllm") is None and not os.path.isfile(
        os.path.join(os.path.dirname(sys.executable), "vllm")
    ):
        uv = shutil.which("uv")
        if uv:
            subprocess.check_call([uv, "pip", "install", "vllm", "--torch-backend=cu128"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    vllm_bin = shutil.which("vllm") or os.path.join(os.path.dirname(sys.executable), "vllm")
    log = open("/tmp/vllm.log", "w")
    subprocess.Popen(
        [vllm_bin, "serve", model, "--port", "8000"],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(90):
        if _vllm_up():
            return
        time.sleep(4)
    tail = open("/tmp/vllm.log").read()[-2000:] if os.path.isfile("/tmp/vllm.log") else ""
    raise RuntimeError(tail or "vLLM did not start on :8000")

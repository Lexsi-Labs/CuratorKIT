"""Map notebook ``BACKEND`` to LiteLLM ``(model, api_base, api_key)``.

vLLM install + ``vllm serve`` live here so Colab run cells stay AlignTune-short.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


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


def _port_busy() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.4):
            return True
    except OSError:
        return False


def _add_nvidia_cuda_libs_to_path() -> None:
    """Load pip nvidia-* libcudart into this process before importing vLLM.

    Copied from AlignTune ``es bug fixes v3`` (8e48d09).
    """
    import ctypes

    dirs = []
    try:
        import nvidia

        root = Path(next(iter(nvidia.__path__)))
        for so in root.rglob("libcudart.so*"):
            dirs.append(so.parent)
    except Exception:
        return
    for d in dict.fromkeys(dirs):
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            path = d / name
            if not path.exists():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
        extra = str(d)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = extra if not current else extra + ":" + current


def _stream_log(log_path: str, seen: int) -> int:
    if not os.path.isfile(log_path):
        return seen
    data = open(log_path).read()
    if len(data) > seen:
        sys.stdout.write(data[seen:])
        sys.stdout.flush()
        return len(data)
    return seen


def _wait_ready(proc: subprocess.Popen | None, log_path: str) -> None:
    """Wait until :8000/v1/models answers. Do not time out a live serve process."""
    seen = 0
    idle = 0
    while True:
        if _vllm_up():
            return
        if proc is not None:
            rc = proc.poll()
            if rc is not None:
                tail = open(log_path).read()[-4000:] if os.path.isfile(log_path) else ""
                raise RuntimeError(tail or f"vLLM exited {rc}")
        new_seen = _stream_log(log_path, seen)
        idle = 0 if new_seen != seen else idle + 1
        seen = new_seen
        if proc is None and idle > 900:
            tail = open(log_path).read()[-4000:] if os.path.isfile(log_path) else ""
            raise RuntimeError(tail or "vLLM did not start on :8000")
        time.sleep(1)


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
    _add_nvidia_cuda_libs_to_path()
    log_path = "/tmp/vllm.log"
    proc: subprocess.Popen | None = None
    if not _port_busy():
        vllm_bin = shutil.which("vllm") or os.path.join(os.path.dirname(sys.executable), "vllm")
        log = open(log_path, "w")
        cmd = [vllm_bin, "serve", model, "--port", "8000"]
        util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
        if util:
            cmd.extend(["--gpu-memory-utilization", util])
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    _wait_ready(proc, log_path)

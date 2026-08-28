"""
HuggingFace Hub publishing with CuratorKIT dataset-card branding.

`brand_hf_repo` creates a Hub **dataset** repo if needed, uploads the packaged
logo, and writes README.md (method / backend / model / tags). Logos ship
inside the package (`curatorkit/assets/*.png`) and are copied into the
destination repo — image srcs point at that repo, never a personal Hugging
Face CDN URL.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CURATORKIT_REPO_URL = "https://github.com/Lexsi-Labs/CuratorKIT"
LEXSI_URL = "https://lexsi.ai/"

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_LOGO_ASSET = _ASSETS_DIR / "curatorkit_logo.png"
_LOGO_REPO_NAME = "curatorkit_logo.png"

FORMAT_FILES: dict[str, str] = {
    "alpaca": "sft_alpaca.jsonl",
    "sharegpt": "sft_sharegpt.jsonl",
    "dpo": "dpo.jsonl",
    "grpo": "grpo.jsonl",
    "corpus": "corpus.jsonl",
    "ppo": "ppo.jsonl",
}

_VALID_KINDS = ("dataset", "format", "rejected")


def _resolve_token(token: Optional[str] = None) -> str:
    token = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        raise ValueError(
            "No HuggingFace token found. Pass token=..., set HF_TOKEN "
            "or HUGGING_FACE_HUB_TOKEN, or run huggingface-cli login."
        )
    return token


def infer_backend(llm_model: str | None) -> str:
    """Map a LiteLLM model string to the notebook BACKEND name."""
    m = (llm_model or "").strip().lower()
    if m.startswith("openrouter/"):
        return "openrouter"
    if m.startswith("vllm/"):
        return "vllm"
    if m.startswith("ollama/"):
        return "ollama"
    if m:
        return "litellm"
    return "none"


def display_model(llm_model: str | None) -> str:
    """Strip the LiteLLM backend prefix for the card table."""
    raw = (llm_model or "").strip()
    for prefix in ("openrouter/", "vllm/", "ollama/"):
        if raw.lower().startswith(prefix):
            return raw[len(prefix):]
    return raw


def hub_meta_from_config(config) -> dict[str, str]:
    """Card fields AlignTune stashes as `_hub_algorithm` / `_hub_backend`.

    Every field is a string. Missing LLM / judge / task never raises —
    hygiene and ingest runs just leave those rows as em dashes.
    """
    model = getattr(config, "llm_model", None) or ""
    judge = getattr(config, "judge_llm_model", None) or ""
    method = getattr(config, "generation_task", None) or "curation"
    return {
        "method": str(method),
        "backend": infer_backend(model) if model else "",
        "model": str(model),
        "judge": str(judge) if judge else "",
    }


def _hf_tag(value: str) -> str:
    s = (value or "").strip().lower().replace(" ", "-").replace("/", "-")
    s = re.sub(r"[^a-z0-9._-]+", "", s).strip("._-")
    return s[:64]


def _task_categories(method: str) -> list[str]:
    m = (method or "").lower()
    cats = ["text-generation"]
    if m in {"qa", "adversarial_qa", "cot"}:
        cats.append("question-answering")
    if m in {"preference", "adversarial_preference"}:
        cats.append("text2text-generation")
    if m in {"grpo", "ppo"}:
        cats.append("reinforcement-learning")
    return cats


def _card_tags(
    method: str,
    backend: str,
    kind: str,
    formats: list[str],
    model: str = "",
) -> list[str]:
    tags = ["curatorkit", "lexsi-labs", "synthetic-data"]
    for raw in (method, backend, kind, display_model(model).rsplit("/", 1)[-1], *formats):
        t = _hf_tag(raw)
        if t and t not in {"none", "-"} and t not in tags:
            tags.append(t)
    return tags


def _hub_asset_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _upload_packaged_asset(api, repo_id: str, token: str, local: Path, name: str) -> Optional[str]:
    if not local.exists():
        logger.warning("Packaged branding asset missing: %s", local)
        return None
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    return _hub_asset_url(repo_id, name)


def _branding_header(logo_url: Optional[str]) -> str:
    if not logo_url:
        return ""
    return f"""<div align="center">
  <table border="0" cellspacing="0" cellpadding="0" style="border: none; border-collapse: collapse;">
    <tr>
      <td align="center" style="border: none; vertical-align: middle;">
        <a href="{LEXSI_URL}"><img src="{logo_url}" alt="CuratorKIT" style="height: 60px; border-radius: 12px;"/></a>
      </td>
    </tr>
  </table>
</div>
"""


def _existing_file(output_dir: Path, name: str) -> Optional[Path]:
    path = Path(output_dir) / name
    if path.is_file() and path.stat().st_size:
        return path
    return None


def _format_paths(output_dir: Path, fmt: Optional[str] = None) -> dict[str, Path]:
    """Export JSONL at the output root, plus `output_split` subdirs if present."""
    wanted = {fmt: FORMAT_FILES[fmt]} if fmt else dict(FORMAT_FILES)
    found: dict[str, Path] = {}

    def _collect(directory: Path, prefix: str = "") -> None:
        for name, filename in wanted.items():
            path = _existing_file(directory, filename)
            if path is None:
                continue
            key = f"{prefix}{name}" if prefix else name
            found[key] = path

    root = Path(output_dir)
    _collect(root)
    for child in sorted(root.iterdir()) if root.is_dir() else []:
        if child.is_dir() and not child.name.startswith("."):
            _collect(child, prefix=f"{child.name}-")
    return found


def render_dataset_card(
    repo_id: str,
    kind: str = "dataset",
    method: str = "",
    backend: str = "",
    model: str = "",
    judge: str = "",
    formats: Optional[list[str]] = None,
    extra_notes: str = "",
    logo_url: Optional[str] = None,
    built_on: str = "",
) -> str:
    """Hub dataset README: YAML tags + method / backend / model table."""
    name = repo_id.split("/")[-1]
    method = method or "curation"
    backend = (backend or "").strip()
    if backend in {"none", "-"}:
        backend = ""
    fmt_list = [f for f in (formats or []) if f]
    fmt_row = ", ".join(f"`{f}`" for f in fmt_list) or "—"
    model_shown = display_model(model) or "—"
    judge_shown = display_model(judge)
    backend_shown = backend or "—"
    header = _branding_header(logo_url)
    stamp = built_on or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cats = "\n".join(f"  - {c}" for c in _task_categories(method))
    tags = "\n".join(f"  - {t}" for t in _card_tags(method, backend, kind, fmt_list, model))
    sample_cfg = fmt_list[0] if fmt_list else None
    load_line = (
        f'ds = load_dataset("{repo_id}", "{sample_cfg}")'
        if sample_cfg
        else f'ds = load_dataset("{repo_id}")'
    )
    judge_row = f"| **Judge** | `{judge_shown}` |\n" if judge_shown else ""
    return f"""---
license: other
pretty_name: {name}
task_categories:
{cats}
tags:
{tags}
---

{header}
# {name}

Built using [CuratorKIT]({CURATORKIT_REPO_URL}) — provenance-grounded curation and synthesis for LLM post-training.

| | |
|---|---|
| **Method** | {method} |
| **Backend** | {backend_shown} |
| **Model** | `{model_shown}` |
{judge_row}| **Formats** | {fmt_row} |
| **Artifact** | {kind} |
| **Published** | {stamp} |

{extra_notes}

## Usage

```python
from datasets import load_dataset

{load_line}
```
"""


def brand_hf_repo(
    repo_id: str,
    kind: str = "dataset",
    task: str = "",
    method: str = "",
    backend: str = "",
    model: str = "",
    judge: str = "",
    formats: Optional[list[str]] = None,
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    """Create the Hub dataset repo if needed, upload the packaged logo, write README.md.

    Always `repo_type="dataset"`. Does not upload JSONL — call after a push helper.
    kind: dataset | format | rejected
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install 'curatorkit[hf]'"
        ) from e

    kind = (kind or "dataset").lower()
    if kind not in _VALID_KINDS:
        raise ValueError(f"kind must be one of {_VALID_KINDS}, got {kind!r}")

    token = _resolve_token(token)
    api = HfApi(token=token)
    built_on = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    logo_url = _upload_packaged_asset(api, repo_id, token, _LOGO_ASSET, _LOGO_REPO_NAME)
    readme = render_dataset_card(
        repo_id,
        kind=kind,
        method=method or task or "curation",
        backend=backend,
        model=model,
        judge=judge,
        formats=formats,
        extra_notes=extra_notes,
        logo_url=logo_url,
        built_on=built_on,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme)
        readme_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    finally:
        os.remove(readme_path)

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Branded %s", url)
    return url


def _brand_safe(
    repo_id: str,
    kind: str = "dataset",
    task: str = "",
    method: str = "",
    backend: str = "",
    model: str = "",
    judge: str = "",
    formats: Optional[list[str]] = None,
    private: bool = False,
    token: Optional[str] = None,
    extra_notes: str = "",
) -> str:
    """Brand the repo; a branding failure does not undo a successful push."""
    try:
        return brand_hf_repo(
            repo_id,
            kind=kind,
            task=task,
            method=method,
            backend=backend,
            model=model,
            judge=judge,
            formats=formats,
            private=private,
            token=token,
            extra_notes=extra_notes,
        )
    except Exception as e:
        logger.warning("Dataset card branding skipped for %s: %s", repo_id, e)
        return f"https://huggingface.co/datasets/{repo_id}"


def _push_jsonl_configs(
    api,
    repo_id: str,
    files: dict[str, Path],
    private: bool,
    token: str,
) -> None:
    from datasets import load_dataset

    for config_name, path in files.items():
        ds = load_dataset("json", data_files=str(path), split="train")
        ds.push_to_hub(
            repo_id,
            config_name=config_name,
            private=private,
            token=token,
        )


def _upload_sidecars(api, repo_id: str, output_dir: Path, token: str) -> None:
    """Upload everything in the run dir except trainer export JSONLs.

    rejected.jsonl, manifest, checksums, probe dumps, pass2, pass3 — if a
    later stage writes a new file, it goes up without editing this list.
    """
    skip = set(FORMAT_FILES.values()) | {_LOGO_REPO_NAME, "README.md"}
    for path in sorted(Path(output_dir).iterdir()):
        if path.name.startswith(".") or not path.is_file() or path.stat().st_size == 0:
            continue
        if path.name in skip:
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )


def _hub_import():
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install 'curatorkit[hf]'"
        ) from e
    return HfApi


def push_output_dir_to_hub(
    output_dir: str | Path,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    task: str = "",
    method: str = "",
    backend: str = "",
    model: str = "",
    judge: str = "",
) -> str:
    """Push every export JSONL in `output_dir`, plus sidecars, then brand.

    Creates a Hub **dataset** repo if it does not already exist.
    """
    HfApi = _hub_import()

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    files = _format_paths(output_dir)
    if not files:
        raise FileNotFoundError(
            f"No export JSONL found in {output_dir}. "
            f"Expected one of: {', '.join(FORMAT_FILES.values())}"
        )

    token = _resolve_token(token)
    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    _push_jsonl_configs(api, repo_id, files, private, token)
    _upload_sidecars(api, repo_id, output_dir, token)
    return _brand_safe(
        repo_id,
        kind="dataset",
        task=task,
        method=method or task,
        backend=backend,
        model=model,
        judge=judge,
        formats=list(files),
        private=private,
        token=token,
    )


def push_format_to_hub(
    output_dir: str | Path,
    repo_id: str,
    fmt: str,
    private: bool = False,
    token: Optional[str] = None,
    task: str = "",
    method: str = "",
    backend: str = "",
    model: str = "",
    judge: str = "",
) -> str:
    """Push a single export format to its own Hub dataset repo."""
    HfApi = _hub_import()

    fmt = fmt.lower().strip()
    if fmt not in FORMAT_FILES:
        raise ValueError(f"Unknown format {fmt!r}. Choose from {list(FORMAT_FILES)}")

    output_dir = Path(output_dir)
    files = _format_paths(output_dir, fmt)
    if not files:
        logger.warning(
            "%s not found in %s — skipping format push for %s",
            FORMAT_FILES[fmt],
            output_dir,
            repo_id,
        )
        return f"https://huggingface.co/datasets/{repo_id}"

    token = _resolve_token(token)
    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    _push_jsonl_configs(api, repo_id, files, private, token)
    return _brand_safe(
        repo_id,
        kind="format",
        task=task,
        method=method or task,
        backend=backend,
        model=model,
        judge=judge,
        formats=[fmt],
        private=private,
        token=token,
    )


def push_rejected_to_hub(
    output_dir: str | Path,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
    task: str = "",
    method: str = "",
    backend: str = "",
    model: str = "",
    judge: str = "",
) -> str:
    """Push rejected.jsonl plus provenance sidecars to a Hub dataset repo."""
    HfApi = _hub_import()

    output_dir = Path(output_dir)
    rejected = _existing_file(output_dir, "rejected.jsonl")
    if rejected is None:
        logger.warning("rejected.jsonl not found in %s — skipping rejected push", output_dir)
        return f"https://huggingface.co/datasets/{repo_id}"

    token = _resolve_token(token)
    api = HfApi(token=token)
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )

    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(rejected), split="train")
    ds.push_to_hub(repo_id, config_name="rejected", private=private, token=token)
    _upload_sidecars(api, repo_id, output_dir, token)
    return _brand_safe(
        repo_id,
        kind="rejected",
        task=task,
        method=method or task,
        backend=backend,
        model=model,
        judge=judge,
        formats=["rejected"],
        private=private,
        token=token,
    )

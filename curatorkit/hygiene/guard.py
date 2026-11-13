"""
GuardModelGate — screen samples with a Hugging Face guard model.

A guard model is a small chat-template classifier that reads a conversation and writes a
verdict: Qwen3Guard-Gen ("Safety: Unsafe / Categories: Violent"), Llama Guard 3/4
("unsafe / S1"). Unlike Detoxify, the multilingual guard models read Korean, Chinese, Swahili
and other languages, so this gate can run as stage 1 for multilingual data, or only on the
fields ToxicityGate could not read (``only_unscreened=True``).

Each field ToxicityGate would score is screened on its own: prompt-side text (instruction,
input, conversation turns) as a user message, response-side text (output, chosen, rejected,
GRPO responses, reference) as an assistant reply to the sample's prompt. Long text is split
into the same 2 048-character windows ToxicityGate uses, so nothing past the model's context is
dropped. The sample is rejected when any window is unsafe (or controversial, by default).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from curatorkit.hygiene.toxicity import ToxicityGate, _resolve_device, fields_to_score
from curatorkit.interfaces import BaseGate
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample

logger = logging.getLogger(__name__)

STEP_VERSION = "1.0.0"

_RESPONSE_FIELDS = ("output", "chosen", "rejected", "reference")
_QWEN = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
_QWEN_CATS = re.compile(r"Categories:\s*(.*)", re.IGNORECASE)


def parse_guard_output(text: str) -> tuple[str | None, list[str]]:
    """``(verdict, categories)`` from a guard model's reply; verdict is ``"safe"``,
    ``"unsafe"``, ``"controversial"`` or ``None`` when the reply has none of them.

    Reads the Qwen3Guard-Gen format (``Safety: Unsafe`` / ``Categories: Violent, PII``) and the
    Llama Guard format (``unsafe`` on the first line, ``S1,S10`` on the next).
    """
    m = _QWEN.search(text)
    if m:
        cats = _QWEN_CATS.search(text)
        names = [c.strip() for c in cats.group(1).split(",")] if cats else []
        return m.group(1).lower(), [c for c in names if c and c.lower() != "none"]
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].lower() in ("safe", "unsafe"):
        cats = [c.strip() for c in lines[1].split(",")] if len(lines) > 1 else []
        return lines[0].lower(), [c for c in cats if c]
    return None, []


class GuardModelGate(BaseGate):
    """
    Reject samples a guard model labels unsafe.

    Parameters
    ----------
    model : str
        Hugging Face id of a chat-template guard model. Default ``Qwen/Qwen3Guard-Gen-0.6B``
        (119 languages, no access request). Llama Guard 3 works the same way once you have
        access; pass ``content_parts=True`` for templates that expect list-of-parts content.
    revision : str | None
        Model revision (commit or tag) to pin; recorded in the config hash.
    text_field : str
        ``"auto"`` screens the same fields as ToxicityGate, or name one DataSample field.
    only_unscreened : bool
        Screen only the fields that an earlier ToxicityGate listed under ``unscreened_fields``
        (text in a script Detoxify cannot read); other samples pass with a ``skipped`` note.
        That list goes by script, so a Latin-script language Detoxify does not read (Swahili) is
        not routed here; leave this off to screen every field.
    reject_controversial : bool
        Reject Qwen3Guard's ``Controversial`` verdict too (default True, the strict reading).
    on_parse_error : "reject" | "pass"
        What to do when the reply has no verdict. Default ``"reject"`` (fail closed).
    batch_size : int
        Conversations per ``generate`` call.
    device : str | None
        ``None`` / ``"auto"``: cuda when available, else cpu. Or ``"cuda:1"``, ``"cpu"``, ...
    max_new_tokens : int
        Tokens the guard may write; a verdict needs about 20.
    content_parts : bool
        Send message content as ``[{"type": "text", "text": ...}]`` (Llama Guard 3-1B / 4 style).
    parser : callable | None
        ``text -> (verdict, categories)`` for a guard with another reply format.
    generator : callable | None
        ``list[list[message]] -> list[str]``: replaces the built-in transformers loader, e.g. a
        vLLM or processor-based model. ``model`` is still recorded as the guard's name.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3Guard-Gen-0.6B",
        *,
        revision: str | None = None,
        text_field: str = "auto",
        only_unscreened: bool = False,
        reject_controversial: bool = True,
        on_parse_error: str = "reject",
        batch_size: int = 8,
        device: str | None = None,
        max_new_tokens: int = 32,
        content_parts: bool = False,
        parser: Callable[[str], tuple[str | None, list[str]]] | None = None,
        generator: Callable[[list[list[dict[str, Any]]]], list[str]] | None = None,
    ) -> None:
        if on_parse_error not in ("reject", "pass"):
            raise ValueError("on_parse_error must be 'reject' or 'pass'")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.revision = revision
        self.text_field = text_field
        self.only_unscreened = only_unscreened
        self.reject_controversial = reject_controversial
        self.on_parse_error = on_parse_error
        self.batch_size = batch_size
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.content_parts = content_parts
        self.parser = parser or parse_guard_output
        self._generator = generator
        self._hf: tuple[Any, Any, str] | None = None

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "revision": self.revision,
                "text_field": self.text_field,
                "only_unscreened": self.only_unscreened,
                "reject_controversial": self.reject_controversial,
                "on_parse_error": self.on_parse_error,
                "max_new_tokens": self.max_new_tokens,
                "content_parts": self.content_parts,
                "custom_parser": self.parser is not parse_guard_output,
                "custom_generator": self._generator is not None,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # -- model ------------------------------------------------------------
    def _load(self) -> tuple[Any, Any, str]:
        if self._hf is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "GuardModelGate needs torch and transformers. Install with: "
                    "pip install 'curatorkit[guard]'"
                ) from e
            device = _resolve_device(self.device)
            tok = AutoTokenizer.from_pretrained(self.model, revision=self.revision)
            tok.padding_side = "left"  # batched generation continues from the right edge
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                self.model, revision=self.revision, dtype=dtype
            ).to(device)
            model.eval()
            self._hf = (tok, model, device)
        return self._hf

    def _generate(self, conversations: list[list[dict[str, Any]]]) -> list[str]:
        """One reply per conversation, greedy, in batches of ``batch_size``."""
        if self._generator is not None:
            return list(self._generator(conversations))
        import torch

        tok, model, device = self._load()
        out: list[str] = []
        for i in range(0, len(conversations), self.batch_size):
            prompts = [
                tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in conversations[i : i + self.batch_size]
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                ids = model.generate(
                    **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            new = ids[:, enc["input_ids"].shape[1] :]  # only the reply, not the prompt
            out.extend(tok.batch_decode(new, skip_special_tokens=True))
        return out

    def _msg(self, role: str, text: str) -> dict[str, Any]:
        content: Any = [{"type": "text", "text": text}] if self.content_parts else text
        return {"role": role, "content": content}

    def _conversation(self, sample: DataSample, field: str, text: str) -> list[dict[str, Any]]:
        if field in _RESPONSE_FIELDS or field.startswith("response_"):
            # A reply is judged in the context of its prompt (response moderation). The prompt is
            # context only, cut to one window; it is screened as its own field where it is scored.
            prompt = "\n\n".join(t for t in (sample.instruction, sample.input) if t.strip())
            if prompt:
                return [self._msg("user", prompt[:2048]), self._msg("assistant", text)]
        return [self._msg("user", text)]

    # -- gate ---------------------------------------------------------------
    @staticmethod
    def _unscreened(sample: DataSample) -> list[str]:
        for rec in reversed(sample.provenance_chain):
            if rec.step_name == "ToxicityGate":
                return list((rec.notes or {}).get("unscreened_fields", []))
        return []

    def _verdict_blocks(self, verdict: str | None) -> bool:
        if verdict is None:
            return self.on_parse_error == "reject"
        return verdict == "unsafe" or (verdict == "controversial" and self.reject_controversial)

    def run(self, samples: list[DataSample]) -> tuple[list[DataSample], list[RejectedSample]]:
        if not samples:
            return [], []
        cfg_hash = self._config_hash()
        ts = datetime.now(UTC)

        def record(sample: DataSample, notes: dict[str, Any]) -> ProvenanceRecord:
            return ProvenanceRecord(
                step_name="GuardModelGate", step_version=STEP_VERSION, timestamp=ts,
                config_hash=cfg_hash, notes={"guard_model": self.model, **notes},
            )

        # (sample index, field, conversation) for every window of every field to screen
        items: list[tuple[int, str, list[dict[str, Any]]]] = []
        screened: dict[int, list[str]] = {}
        for i, sample in enumerate(samples):
            pairs = fields_to_score(sample, self.text_field)
            if self.only_unscreened:
                wanted = set(self._unscreened(sample))
                pairs = [(f, t) for f, t in pairs if f in wanted]
            screened[i] = [f for f, _ in pairs]
            for field, text in pairs:
                for window in ToxicityGate._windows(text):
                    items.append((i, field, self._conversation(sample, field, window)))

        t0 = datetime.now(UTC)
        replies = self._generate([conv for _, _, conv in items]) if items else []
        seconds = (datetime.now(UTC) - t0).total_seconds()

        # worst verdict per (sample, field): blocking beats unparsed beats safe
        per_sample: dict[int, dict[str, dict[str, Any]]] = {}
        for (i, field, _), reply in zip(items, replies, strict=True):
            verdict, cats = self.parser(reply)
            cur = per_sample.setdefault(i, {}).get(field)
            entry = {"verdict": verdict, "categories": cats, "raw": reply.strip()[:200]}
            if cur is None or (self._verdict_blocks(verdict) and not self._verdict_blocks(cur["verdict"])):
                per_sample[i][field] = entry

        passed: list[DataSample] = []
        rejected: list[RejectedSample] = []
        for i, sample in enumerate(samples):
            if not screened[i]:
                reason = "no_unscreened_fields" if self.only_unscreened else "no_text"
                sample.append_provenance(record(sample, {"passed": True, "skipped": True, "reason": reason}))
                passed.append(sample)
                continue
            fields = per_sample.get(i, {})
            blocking = [f for f, e in fields.items() if self._verdict_blocks(e["verdict"])]
            notes = {
                "phase": "guard_model",
                "screened_fields": screened[i],
                "field_verdicts": {f: {"verdict": e["verdict"], "categories": e["categories"]}
                                   for f, e in fields.items()},
            }
            if blocking:
                worst = fields[blocking[0]]
                label = worst["verdict"] or "unparsed"
                rej = RejectedSample(
                    **sample.model_dump(),
                    rejection_reason=f"unsafe_content:guard_model:{label}",
                    rejecting_step="GuardModelGate",
                )
                rej.append_provenance(record(sample, {
                    "passed": False, **notes, "worst_field": blocking[0],
                    "categories": worst["categories"], "raw": worst["raw"],
                }))
                rejected.append(rej)
            else:
                sample.append_provenance(record(sample, {"passed": True, **notes}))
                passed.append(sample)

        logger.info(
            "GuardModelGate (%s): %d windows from %d samples in %.1f s; rejected %d",
            self.model, len(items), len(samples), seconds, len(rejected),
        )
        return passed, rejected

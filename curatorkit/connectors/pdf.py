"""
PDF connector — wraps MinerU for layout parsing, implements all chunking logic.

PDFReader handles file I/O and MinerU invocation.
PDFChunker handles all splitting decisions — they are separate so chunking
strategy can be tested in isolation without a real PDF file.

Optional features:
  - extract_tables=True extracts table content instead of skipping tables.
    Tables become DataSample objects with task_type="table_qa" and the
    table content stored in metadata["table_content"].
  - output_mode controls what happens per chunk after extraction:
      "chunk"      — raw text only (default, no LLM call)
      "qa"         — LLM generates question + grounded answer
      "preference" — LLM generates chosen + rejected pair
      "grpo"       — LLM generates N responses + scores
      "multiturn"  — LLM extends chunk into multi-turn dialogue
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Matches a sentence boundary: punctuation followed by whitespace followed by
# a capital letter. Avoids splitting on decimal numbers (3.14), abbreviations
# (U.S.), and lowercase continuations — all of which lack the capital lookahead.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

from curatorkit.interfaces import BaseReader
from curatorkit.schema import DataSample, ProvenanceRecord, RejectedSample
from curatorkit.utils.tokens import count_tokens_whitespace

STEP_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# PDFChunker — all splitting logic lives here, testable without a PDF
# ---------------------------------------------------------------------------


class PDFChunker:
    """Split MinerU block output into token-bounded chunks.

    Args:
        strategy: 'heading' | 'sentence' | 'fixed'
        max_tokens: Maximum tokens per chunk (whitespace tokenizer).
        overlap_tokens: Tokens from end of chunk N prepended to chunk N+1.
        extract_tables: If True, include table content as chunks instead
                        of emitting stubs.
    """

    def __init__(
        self,
        strategy: str = "heading",
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        extract_tables: bool = False,
        min_section_tokens: int = 30,
    ) -> None:
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.extract_tables = extract_tables
        self.min_section_tokens = min_section_tokens

    def chunk(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a list of chunk dicts with keys: text, page, heading, chunk_index."""
        if self.strategy == "heading":
            return self._chunk_by_heading(blocks)
        elif self.strategy == "sentence":
            return self._chunk_by_sentence(blocks)
        else:
            return self._chunk_fixed(blocks)

    def _chunk_by_heading(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current_heading = None
        current_page = 1
        buffer: list[str] = []

        for block in blocks:
            btype = block.get("type", "paragraph")
            text = block.get("text", "").strip()
            page = block.get("page", current_page)

            if btype == "heading":
                if buffer:
                    sections.append(
                        {
                            "heading": current_heading,
                            "page": current_page,
                            "text": " ".join(buffer),
                        }
                    )
                    buffer = []
                current_heading = text
                current_page = page
            elif btype == "table":
                # Extract table content if enabled
                if self.extract_tables and text:
                    # Flush current buffer first
                    if buffer:
                        sections.append(
                            {
                                "heading": current_heading,
                                "page": current_page,
                                "text": " ".join(buffer),
                            }
                        )
                        buffer = []
                    # Emit table as its own chunk
                    table_html = block.get("html", "")
                    sections.append(
                        {
                            "heading": current_heading,
                            "page": page,
                            "text": text,
                            "_is_table": True,
                            "_table_html": table_html,
                            "_table_bbox": block.get("bbox", []),
                        }
                    )
                else:
                    # Table extraction disabled — record as skipped
                    sections.append(
                        {
                            "heading": current_heading,
                            "page": page,
                            "text": "",
                            "_table_skipped": True,
                        }
                    )
            else:
                if text:
                    buffer.append(text)
                    current_page = page

        if buffer:
            sections.append(
                {
                    "heading": current_heading,
                    "page": current_page,
                    "text": " ".join(buffer),
                }
            )

        # Merge orphan sections (< min_section_tokens) into the next section.
        # Prevents single-sentence stubs like "Reserved." from becoming their
        # own chunk, which dilutes downstream QA generation quality.
        merged: list[dict[str, Any]] = []
        carry_text = ""
        carry_heading = None
        carry_page = 1
        for sec in sections:
            if sec.get("_table_skipped") or sec.get("_is_table"):
                if carry_text:
                    merged.append(
                        {"heading": carry_heading, "page": carry_page, "text": carry_text}
                    )
                    carry_text = ""
                merged.append(sec)
                continue
            tok_count = count_tokens_whitespace(sec["text"])
            if tok_count < self.min_section_tokens:
                # Accumulate into carry buffer
                if not carry_text:
                    carry_heading = sec["heading"]
                    carry_page = sec["page"]
                carry_text = (carry_text + " " + sec["text"]).strip()
            else:
                if carry_text:
                    sec = dict(sec)
                    sec["text"] = (carry_text + " " + sec["text"]).strip()
                    carry_text = ""
                merged.append(sec)
        if carry_text:
            merged.append({"heading": carry_heading, "page": carry_page, "text": carry_text})

        return self._apply_token_cap_and_overlap(merged)

    def _chunk_by_sentence(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        full_text = " ".join(
            b.get("text", "").strip()
            for b in blocks
            if b.get("type") not in ("table", "figure") and b.get("text", "").strip()
        )
        sentences = [s for s in _SENTENCE_BOUNDARY.split(full_text) if s.strip()]

        chunks: list[dict[str, Any]] = []
        buffer: list[str] = []

        for sent in sentences:
            if count_tokens_whitespace(" ".join(buffer + [sent])) > self.max_tokens:
                if buffer:
                    chunks.append({"text": " ".join(buffer), "page": 1, "heading": None})
                buffer = [sent]
            else:
                buffer.append(sent)

        if buffer:
            chunks.append({"text": " ".join(buffer), "page": 1, "heading": None})

        # Also extract tables if enabled
        if self.extract_tables:
            for b in blocks:
                if b.get("type") == "table" and b.get("text", "").strip():
                    chunks.append(
                        {
                            "text": b["text"].strip(),
                            "page": b.get("page", 1),
                            "heading": None,
                            "_is_table": True,
                            "_table_html": b.get("html", ""),
                            "_table_bbox": b.get("bbox", []),
                        }
                    )

        return self._add_overlap(chunks)

    def _chunk_fixed(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_tokens = " ".join(
            b.get("text", "").strip() for b in blocks if b.get("text", "").strip()
        ).split()

        chunks: list[dict[str, Any]] = []
        stride = max(1, self.max_tokens - self.overlap_tokens)
        i = 0
        while i < len(all_tokens):
            window = all_tokens[i : i + self.max_tokens]
            chunks.append({"text": " ".join(window), "page": 1, "heading": None})
            i += stride

        return chunks

    def _apply_token_cap_and_overlap(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split over-length heading sections at sentence boundaries, then add overlap."""
        result: list[dict[str, Any]] = []
        for sec in sections:
            if sec.get("_table_skipped") or sec.get("_is_table"):
                result.append(sec)
                continue
            text = sec["text"]
            if count_tokens_whitespace(text) <= self.max_tokens:
                result.append(sec)
            else:
                sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
                buffer: list[str] = []
                for sent in sentences:
                    candidate = " ".join(buffer + [sent])
                    if count_tokens_whitespace(candidate) > self.max_tokens and buffer:
                        result.append(
                            {
                                "text": " ".join(buffer),
                                "page": sec["page"],
                                "heading": sec["heading"],
                            }
                        )
                        buffer = [sent]
                    else:
                        buffer.append(sent)
                if buffer:
                    result.append(
                        {
                            "text": " ".join(buffer),
                            "page": sec["page"],
                            "heading": sec["heading"],
                        }
                    )
        return self._add_overlap(result)

    def _add_overlap(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepend overlap_tokens tokens from chunk N to chunk N+1."""
        if self.overlap_tokens == 0:
            return chunks
        out = []
        for i, chunk in enumerate(chunks):
            if chunk.get("_table_skipped") or chunk.get("_is_table"):
                out.append(chunk)
                continue
            if (
                i > 0
                and not chunks[i - 1].get("_table_skipped")
                and not chunks[i - 1].get("_is_table")
            ):
                prev_tokens = chunks[i - 1]["text"].split()[-self.overlap_tokens :]
                text = " ".join(prev_tokens) + " " + chunk["text"]
                out.append({**chunk, "text": text.strip()})
            else:
                out.append(chunk)
        return out


# ---------------------------------------------------------------------------
# PDFReader — file I/O, MinerU invocation, and output_mode dispatch
# ---------------------------------------------------------------------------


class PDFReader(BaseReader):
    """Read a PDF via MinerU and produce chunked DataSamples.

    MinerU must be installed separately:
      pip install 'curatorkit[pdf]'

    Optional parameters:
      extract_tables  — If True, extract table text as real chunks
                        instead of emitting stubs.
      output_mode     — "chunk" (default) | "qa" | "preference"
                        | "grpo" | "multiturn". Modes other than "chunk"
                        trigger LLM generation per chunk (requires llm_model).
      llm_model       — LiteLLM model string for generation modes.
    """

    def __init__(
        self,
        path: Path,
        chunk_strategy: str = "heading",
        chunk_max_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        extract_tables: bool = False,
        ocr: bool = False,
        min_section_tokens: int = 30,
        output_mode: str = "chunk",
        llm_model: str | None = None,
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1024,
        llm_api_key: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.chunk_strategy = chunk_strategy
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.extract_tables = extract_tables
        self.ocr = ocr
        self.min_section_tokens = min_section_tokens
        self.output_mode = output_mode
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_api_key = llm_api_key

        self.chunker = PDFChunker(
            strategy=chunk_strategy,
            max_tokens=chunk_max_tokens,
            overlap_tokens=chunk_overlap_tokens,
            extract_tables=extract_tables,
            min_section_tokens=min_section_tokens,
        )

        # Validate output_mode requires LLM
        if output_mode != "chunk" and not llm_model:
            raise ValueError(
                f"output_mode='{output_mode}' requires llm_model to be set. "
                f"Use output_mode='chunk' to chunk documents without an LLM."
            )

        # Verify mineru is installed
        try:
            import mineru  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "\nMinerU is not installed. Fix:\n\n"
                '    pip install "curatorkit[pdf]"\n\n'
                "A CUDA GPU is used automatically when available (recommended "
                "for speed); CPU works but is slower.\n"
                "Model weights are downloaded automatically on first use."
            ) from e

        # Set device preference: CUDA first
        import os

        if "MINERU_DEVICE_MODE" not in os.environ:
            try:
                import torch

                if torch.cuda.is_available():
                    os.environ["MINERU_DEVICE_MODE"] = "cuda"
            except ImportError:
                pass

    def _config_hash(self) -> str:
        payload = json.dumps(
            {
                "chunk_strategy": self.chunk_strategy,
                "chunk_max_tokens": self.chunk_max_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
                "ocr": self.ocr,
                "extract_tables": self.extract_tables,
                "output_mode": self.output_mode,
                "llm_model": self.llm_model,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def read(self) -> tuple[list[DataSample], list[RejectedSample]]:
        blocks = self._extract_blocks()
        chunks = self.chunker.chunk(blocks)
        samples: list[DataSample] = []
        rejected: list[RejectedSample] = []
        cfg_hash = self._config_hash()
        chunk_idx = 0

        for chunk in chunks:
            # ── Table stub (extract_tables=False) ────────────────────────
            if chunk.get("_table_skipped"):
                r = RejectedSample(
                    source_uri=str(self.path),
                    instruction="",
                    rejection_reason="table_stubbed:extract_tables_disabled",
                    rejecting_step="PDFReader",
                    metadata={"page": chunk.get("page")},
                )
                r.append_provenance(
                    ProvenanceRecord(
                        step_name="PDFReader",
                        step_version=STEP_VERSION,
                        timestamp=datetime.now(UTC).replace(tzinfo=None),
                        config_hash=cfg_hash,
                        notes={
                            "table_extraction": "skipped",
                            "reason": "extract_tables_disabled",
                            "page": chunk.get("page"),
                            "source_file": str(self.path),
                        },
                    )
                )
                rejected.append(r)
                continue

            text = chunk.get("text", "").strip()
            if not text:
                continue

            # ── Table chunk (extract_tables=True) ────────────────────────
            is_table = chunk.get("_is_table", False)

            if is_table:
                task_type = "table_qa" if self.output_mode != "chunk" else "language_modeling"
                sample = DataSample(
                    source_uri=str(self.path),
                    instruction="" if self.output_mode == "chunk" else text,
                    output=text if self.output_mode == "chunk" else "",
                    task_type=task_type,
                    metadata={
                        "page": chunk.get("page"),
                        "parent_heading": chunk.get("heading"),
                        "chunk_index": chunk_idx,
                        "content_type": "table",
                        "table_html": chunk.get("_table_html", ""),
                        "table_bbox": chunk.get("_table_bbox", []),
                        "source_file": str(self.path),
                    },
                )
            else:
                # ── Text chunk ───────────────────────────────────────────
                task_type = "language_modeling"
                sample = DataSample(
                    source_uri=str(self.path),
                    output=text,
                    task_type=task_type,
                    metadata={
                        "page": chunk.get("page"),
                        "parent_heading": chunk.get("heading"),
                        "chunk_index": chunk_idx,
                        "content_type": "text",
                        "source_file": str(self.path),
                    },
                )

            sample.append_provenance(
                ProvenanceRecord(
                    step_name="PDFReader",
                    step_version=STEP_VERSION,
                    timestamp=datetime.now(UTC).replace(tzinfo=None),
                    config_hash=cfg_hash,
                    notes={
                        "source_file": str(self.path),
                        "page": chunk.get("page"),
                        "chunk_index": chunk_idx,
                        "parent_heading": chunk.get("heading"),
                        "chunk_strategy": self.chunk_strategy,
                        "overlap_tokens": self.chunk_overlap_tokens,
                        "ocr": self.ocr,
                        "is_table": is_table,
                        "output_mode": self.output_mode,
                    },
                )
            )
            samples.append(sample)
            chunk_idx += 1

        # ═════════════════════════════════════════════════════════════════════
        # output_mode LLM generation
        # ═════════════════════════════════════════════════════════════════════
        if self.output_mode != "chunk" and self.llm_model and samples:
            samples, gen_rejected = self._apply_generation(samples)
            rejected.extend(gen_rejected)

        return samples, rejected

    def _apply_generation(
        self, samples: list[DataSample]
    ) -> tuple[list[DataSample], list[RejectedSample]]:
        """Apply LLM generation based on output_mode."""
        llm = self._build_llm()
        mode = self.output_mode

        if mode == "qa":
            from curatorkit.generators.qa_generator import QAGenerationTask

            task = QAGenerationTask(llm=llm, num_questions=3)
        elif mode == "preference":
            from curatorkit.generators.preference_gen import PreferenceGenerationTask

            task = PreferenceGenerationTask(llm=llm)
        elif mode == "grpo":
            from curatorkit.generators.grpo_rollout import GRPORolloutTask

            task = GRPORolloutTask(llm=llm, num_responses=4)
        elif mode == "multiturn":
            from curatorkit.generators.multiturn_gen import MultiTurnTask

            task = MultiTurnTask(llm=llm, num_turns=3)
        else:
            return samples, []

        generated = task.run(samples)
        return generated, task.flush_rejected()

    def _build_llm(self):
        """Build an LLM backend for generation modes."""
        model = self.llm_model
        if model.startswith("ollama/") or model.startswith("ollama_chat/"):
            from curatorkit.llm.ollama import OllamaBackend

            return OllamaBackend(
                model=model.split("/", 1)[1],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
        from curatorkit.llm.litellm import LiteLLMBackend

        return LiteLLMBackend(
            model=model,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            api_key=self.llm_api_key,
        )

    def _extract_blocks(self) -> list[dict[str, Any]]:
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
        from mineru.data.data_reader_writer import FileBasedDataWriter

        pdf_bytes = self.path.read_bytes()
        local_image_dir = self.path.parent / "images"
        local_image_dir.mkdir(exist_ok=True)
        image_writer = FileBasedDataWriter(str(local_image_dir))

        parse_method = "ocr" if self.ocr else "auto"
        result_holder: dict = {}

        def on_doc_ready(doc_index, model_list, middle_json, ocr_enable):
            result_holder["middle_json"] = middle_json

        doc_analyze_streaming(
            pdf_bytes_list=[pdf_bytes],
            image_writer_list=[image_writer],
            lang_list=["en"],
            on_doc_ready=on_doc_ready,
            parse_method=parse_method,
            formula_enable=False,
            table_enable=self.extract_tables,
        )

        pdf_info = result_holder["middle_json"]["pdf_info"]
        return self._mineru_to_blocks(pdf_info)

    def _mineru_to_blocks(self, pdf_info: list) -> list[dict[str, Any]]:
        """Convert MinerU pdf_info structure to our internal block format.

        pdf_info is a list of page dicts, each with 'para_blocks'.
        """
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
        from mineru.utils.enum_class import BlockType, ContentType

        blocks: list[dict[str, Any]] = []
        for page_no, page_data in enumerate(pdf_info):
            page_num = page_no + 1
            for para_block in page_data.get("para_blocks", []):
                btype = para_block.get("type", "")

                if btype == BlockType.TITLE:
                    text = merge_para_with_text(para_block).strip()
                    if text:
                        blocks.append({"type": "heading", "text": text, "page": page_num})

                elif btype == BlockType.TEXT:
                    text = merge_para_with_text(para_block).strip()
                    if text:
                        blocks.append({"type": "paragraph", "text": text, "page": page_num})

                elif btype == BlockType.TABLE:
                    table_text = ""
                    table_html = ""
                    bbox = para_block.get("bbox", [])
                    for sub in para_block.get("blocks", []):
                        if sub.get("type") == BlockType.TABLE_BODY:
                            for line in sub.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("type") == ContentType.TABLE:
                                        table_html = span.get("latex", "") or span.get("html", "")
                                        table_text = span.get("text", "") or table_html
                        elif sub.get("type") in (BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE):
                            caption = merge_para_with_text(sub).strip()
                            if caption:
                                table_text = (table_text + " " + caption).strip()
                    blocks.append(
                        {
                            "type": "table",
                            "text": table_text,
                            "html": table_html,
                            "bbox": bbox,
                            "page": page_num,
                        }
                    )

                # Skip Image, InterlineEquation, figure captions — no text content for chunking

        return blocks

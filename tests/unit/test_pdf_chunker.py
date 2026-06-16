"""Unit tests for PDFChunker — strategy testing without requiring a PDF file."""

from __future__ import annotations

from curatorkit.connectors.pdf import PDFChunker


def make_blocks(texts: list[str], types: list[str] | None = None) -> list[dict]:
    types = types or ["paragraph"] * len(texts)
    return [
        {"type": t, "text": text, "page": i + 1} for i, (t, text) in enumerate(zip(types, texts))
    ]


class TestHeadingChunker:
    def test_splits_at_headings(self):
        blocks = make_blocks(
            ["Introduction", "Some intro text here.", "Methods", "We used Python."],
            ["heading", "paragraph", "heading", "paragraph"],
        )
        # min_section_tokens=1 disables orphan-merging so these short
        # sections stay separate (the default merges sections < 30 tokens
        # into the next one).
        chunker = PDFChunker(
            strategy="heading",
            max_tokens=512,
            overlap_tokens=0,
            min_section_tokens=1,
        )
        chunks = chunker.chunk(blocks)
        # Should produce 2 heading sections
        texts = [c["text"] for c in chunks if not c.get("_table_skipped")]
        assert len(texts) == 2

    def test_table_produces_skipped_record(self):
        blocks = [
            {"type": "heading", "text": "Results", "page": 1},
            {"type": "table", "text": "Table data", "page": 1},
            {"type": "paragraph", "text": "Discussion text here.", "page": 1},
        ]
        chunker = PDFChunker(strategy="heading", max_tokens=512, overlap_tokens=0)
        chunks = chunker.chunk(blocks)
        skipped = [c for c in chunks if c.get("_table_skipped")]
        assert len(skipped) == 1

    def test_long_section_split_at_sentence(self):
        # Create a section that exceeds max_tokens
        long_paragraph = ". ".join([f"Sentence number {i} with some words" for i in range(30)])
        blocks = [
            {"type": "heading", "text": "Long Section", "page": 1},
            {"type": "paragraph", "text": long_paragraph, "page": 1},
        ]
        chunker = PDFChunker(strategy="heading", max_tokens=20, overlap_tokens=0)
        chunks = chunker.chunk(blocks)
        non_skipped = [c for c in chunks if not c.get("_table_skipped")]
        assert len(non_skipped) > 1

    def test_overlap_prepended(self):
        blocks = [
            {"type": "heading", "text": "Section A", "page": 1},
            {"type": "paragraph", "text": "alpha beta gamma delta", "page": 1},
            {"type": "heading", "text": "Section B", "page": 2},
            {"type": "paragraph", "text": "epsilon zeta", "page": 2},
        ]
        # min_section_tokens=1 keeps the two short sections separate so the
        # overlap behaviour is actually exercised.
        chunker = PDFChunker(
            strategy="heading",
            max_tokens=512,
            overlap_tokens=2,
            min_section_tokens=1,
        )
        chunks = chunker.chunk(blocks)
        non_skipped = [c for c in chunks if not c.get("_table_skipped")]
        if len(non_skipped) >= 2:
            # Second chunk should start with overlap from first
            assert "delta" in non_skipped[1]["text"] or "gamma" in non_skipped[1]["text"]


class TestFixedChunker:
    def test_fixed_chunking(self):
        words = " ".join([f"word{i}" for i in range(100)])
        blocks = [{"type": "paragraph", "text": words, "page": 1}]
        chunker = PDFChunker(strategy="fixed", max_tokens=20, overlap_tokens=5)
        chunks = chunker.chunk(blocks)
        assert len(chunks) > 1
        for chunk in chunks:
            token_count = len(chunk["text"].split())
            assert token_count <= 25  # max_tokens + some overlap tolerance


class TestSentenceChunker:
    def test_sentence_chunking(self):
        text = ". ".join([f"This is sentence number {i}" for i in range(20)])
        blocks = [{"type": "paragraph", "text": text, "page": 1}]
        chunker = PDFChunker(strategy="sentence", max_tokens=10, overlap_tokens=0)
        chunks = chunker.chunk(blocks)
        assert len(chunks) > 1

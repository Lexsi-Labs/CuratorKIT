"""Unit tests for TextCleaner — each transform tested independently."""

from __future__ import annotations

from curatorkit.normalizers.clean import TextCleaner
from curatorkit.schema import DataSample


def make_sample(
    instruction: str = "Hello world", output: str = "A valid answer here"
) -> DataSample:
    return DataSample(source_uri="test://", instruction=instruction, output=output)


class TestStripHtml:
    def test_removes_tags(self):
        cleaner = TextCleaner(
            transforms={
                "strip_html": True,
                "normalise_unicode": False,
                "fix_encoding_artifacts": False,
                "collapse_whitespace": False,
                "remove_control_chars": False,
            }
        )
        sample = make_sample(instruction="<b>Hello</b> <em>world</em>")
        result = cleaner.run([sample])
        assert result[0].instruction == "Hello world"

    def test_no_tags_unchanged(self):
        cleaner = TextCleaner()
        sample = make_sample(instruction="Plain text no tags here")
        result = cleaner.run([sample])
        assert "Plain text" in result[0].instruction


class TestNormaliseUnicode:
    def test_nfc_normalisation(self):
        # é can be encoded as U+00E9 (NFC) or U+0065 U+0301 (NFD)
        nfd_text = "cafe\u0301"  # NFD: e + combining acute
        nfc_expected = "caf\u00e9"  # NFC: é as single codepoint
        cleaner = TextCleaner()
        sample = make_sample(instruction=nfd_text)
        result = cleaner.run([sample])
        assert result[0].instruction == nfc_expected


class TestCollapseWhitespace:
    def test_collapses_spaces(self):
        cleaner = TextCleaner()
        sample = make_sample(instruction="Hello   world   how  are you")
        result = cleaner.run([sample])
        assert "  " not in result[0].instruction

    def test_preserves_newlines(self):
        cleaner = TextCleaner()
        sample = make_sample(instruction="Line one\nLine two\nLine three")
        result = cleaner.run([sample])
        assert "\n" in result[0].instruction


class TestRemoveControlChars:
    def test_removes_control_chars(self):
        cleaner = TextCleaner()
        sample = make_sample(instruction="Hello\x01\x02\x03World")
        result = cleaner.run([sample])
        assert "\x01" not in result[0].instruction
        assert "HelloWorld" in result[0].instruction

    def test_preserves_tab_and_newline_from_control_char_removal(self):
        # run only remove_control_chars — collapse_whitespace intentionally converts tabs to spaces
        cleaner = TextCleaner(
            transforms={
                "strip_html": False,
                "normalise_unicode": False,
                "fix_encoding_artifacts": False,
                "collapse_whitespace": False,
                "remove_control_chars": True,
            }
        )
        sample = make_sample(instruction="Col1\tCol2\nRow2")
        result = cleaner.run([sample])
        assert "\t" in result[0].instruction  # tab (0x09) is NOT a control char
        assert "\n" in result[0].instruction  # newline (0x0A) is NOT a control char


class TestToggleableTransforms:
    def test_can_disable_html_strip(self):
        cleaner = TextCleaner(transforms={"strip_html": False})
        sample = make_sample(instruction="<b>Bold</b>")
        result = cleaner.run([sample])
        assert "<b>" in result[0].instruction

    def test_provenance_records_applied_transforms(self):
        cleaner = TextCleaner(
            transforms={
                "strip_html": True,
                "collapse_whitespace": False,
                "normalise_unicode": False,
                "fix_encoding_artifacts": False,
                "remove_control_chars": False,
            }
        )
        sample = make_sample()
        result = cleaner.run([sample])
        notes = result[0].provenance_chain[-1].notes
        assert "strip_html" in notes["transforms_applied"]
        assert "collapse_whitespace" not in notes["transforms_applied"]

from __future__ import annotations

from pathlib import Path

from peacock_asr.papers.convert import (
    QCResult,
    _ensure_title_heading,
    _extract_title,
    _is_better_qc,
    evaluate_markdown_quality,
    normalize_markdown,
    parse_arxiv_id,
    promote_section_headings,
)


def test_parse_arxiv_id_from_filename() -> None:
    path = Path("2204.03067_byt5_massively_multilingual_g2p.pdf")
    assert parse_arxiv_id(path) == "2204.03067"


def test_parse_arxiv_id_absent() -> None:
    path = Path("rezackova21_interspeech_t5g2p.pdf")
    assert parse_arxiv_id(path) is None


def test_normalize_markdown_joins_hyphenated_breaks() -> None:
    text = "Line with hy-\nphenated word.\n\n\nAnother line.\u00ad"
    normalized = normalize_markdown(text)
    assert "hyphenated" in normalized
    assert "\u00ad" not in normalized
    assert "\n\n\n" not in normalized


def test_normalize_markdown_cleans_footnote_markers() -> None:
    text = (
        "See details.111https://example.org and 222Previous work and "
        "1All the transcriptions."
    )
    normalized = normalize_markdown(text)
    assert "111https://" not in normalized
    assert "222Previous" not in normalized
    assert "1All the transcriptions" not in normalized
    assert "https://example.org" in normalized
    assert "Previous work" in normalized
    assert "All the transcriptions" in normalized


def test_quality_check_reports_warnings() -> None:
    markdown = "# Title\n\nShort text without marker.\n"
    qc = evaluate_markdown_quality(markdown, min_words=50)
    assert not qc.pass_qc
    assert qc.word_count < 50
    assert not qc.has_abstract
    assert qc.warnings


def test_ensure_title_heading_inserts_heading() -> None:
    markdown = "A Title\n\nAbstract\n\nBody text."
    result = _ensure_title_heading(markdown=markdown, title="A Title")
    assert result.startswith("# A Title")


def test_ensure_title_heading_rewrites_and_dedupes_wrapped_heading() -> None:
    markdown = (
        "# Short Title\n"
        "Byte-to-Byte Models\n\n"
        "###### Abstract\n"
    )
    title = "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
    result = _ensure_title_heading(markdown=markdown, title=title)
    assert result.startswith(f"# {title}")
    assert "\nByte-to-Byte Models\n" not in result


def test_ensure_title_heading_dedupes_repeated_title_line() -> None:
    title = "T5G2P: Using Text-to-Text Transfer Transformer"
    markdown = (
        f"# {title}\n"
        f"{title}\n"
        "###### Abstract\n"
    )
    result = _ensure_title_heading(markdown=markdown, title=title)
    assert result.count(title) == 1


def test_ensure_title_heading_dedupes_repeated_title_line_after_prepend() -> None:
    title = "T5G2P: Using Text-to-Text Transfer Transformer"
    markdown = (
        f"{title}\n"
        "for Grapheme-to-Phoneme Conversion\n"
        "Abstract\n"
    )
    result = _ensure_title_heading(markdown=markdown, title=title)
    assert result.count(title) == 1


def test_ensure_title_heading_dedupes_repeated_split_title_after_prepend() -> None:
    title = (
        "T5G2P: Using Text-to-Text Transfer Transformer "
        "for Grapheme-to-Phoneme Conversion"
    )
    markdown = (
        "T5G2P: Using Text-to-Text Transfer Transformer\n"
        "for Grapheme-to-Phoneme Conversion\n"
        "Marketa Rezackova\n"
    )
    result = _ensure_title_heading(markdown=markdown, title=title)
    assert "T5G2P: Using Text-to-Text Transfer Transformer\n" not in result
    assert "\nfor Grapheme-to-Phoneme Conversion\n" not in result


def test_promote_section_headings() -> None:
    markdown = "# Title\n\nAbstract\n\n1 Introduction\n\nBody."
    result = promote_section_headings(markdown)
    assert "## Abstract" in result
    assert "## 1 Introduction" in result


def test_promote_dotted_headings_and_abstract_prefix() -> None:
    markdown = "# Title\n\nAbstract      body text\n\n2. Method\n\nBody."
    result = promote_section_headings(markdown)
    assert "## Abstract" in result
    assert "## 2 Method" in result


def test_promote_roman_heading_with_trailing_text() -> None:
    markdown = "# Title\n\nI. INTRODUCTION      right-column content\n\nBody."
    result = promote_section_headings(markdown)
    assert "## I INTRODUCTION" in result
    assert "right-column content" in result


def test_promote_heading_with_wide_internal_spacing() -> None:
    markdown = "# Title\n\n1       Introduction\n\nBody."
    result = promote_section_headings(markdown)
    assert "## 1 Introduction" in result


def test_promote_section_headings_does_not_promote_long_numbered_sentence() -> None:
    markdown = (
        "# Title\n\n"
        "1 LoRA-only fine-tuning: Updating only the weights of the LoRA "
        "adapter layers, "
        "while keeping other layers frozen.\n\nBody."
    )
    result = promote_section_headings(markdown)
    assert "## 1 LoRA-only fine-tuning" not in result


def test_promote_section_headings_does_not_promote_transcript_numbered_line() -> None:
    markdown = '# Title\n\n1 Transcript: "hello there."\n\nBody.'
    result = promote_section_headings(markdown)
    assert '## 1 Transcript: "hello there."' not in result


def test_promote_section_headings_demotes_overlong_existing_heading() -> None:
    markdown = (
        "# Title\n\n"
        "## Transcript: he might be seeking validation from his friends, or maybe "
        "he is not aware of how his behavior affects you.\n\nBody."
    )
    result = promote_section_headings(markdown)
    assert "## Transcript:" not in result
    assert "Transcript: he might be seeking validation" in result


def test_is_better_qc_prefers_fewer_warnings() -> None:
    baseline = QCResult(pass_qc=False, warnings=["w1", "w2"], word_count=100)
    candidate = QCResult(pass_qc=False, warnings=["w1"], word_count=80)
    assert _is_better_qc(candidate=candidate, baseline=baseline)


def test_quality_check_accepts_long_preface_without_abstract_marker() -> None:
    preface = " ".join(["preface"] * 140)
    markdown = f"# Title\n\n{preface}\n\n## 1 Introduction\n\nBody text."
    qc = evaluate_markdown_quality(markdown, min_words=50)
    assert qc.has_abstract


def test_extract_title_skips_conference_header() -> None:
    markdown = (
        "INTERSPEECH 2021\n"
        "Interspeech 2021\n"
        "A Better Grapheme-to-Phoneme Model for Low-Resource Languages\n"
    )
    assert _extract_title(markdown) == (
        "A Better Grapheme-to-Phoneme Model for Low-Resource Languages"
    )


def test_extract_title_skips_date_line() -> None:
    markdown = (
        "30 August - 3 September, 2021, Brno, Czechia\n"
        "T5G2P: Using Text-to-Text Transfer Transformer\n"
    )
    assert _extract_title(markdown) == "T5G2P: Using Text-to-Text Transfer Transformer"


def test_extract_title_joins_wrapped_title_line() -> None:
    markdown = (
        "ByT5: Towards a Token-Free Future with Pre-trained\n"
        "Byte-to-Byte Models\n"
        "Abstract\n"
    )
    assert _extract_title(markdown) == (
        "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
    )


def test_extract_title_joins_wrapped_heading_title_line() -> None:
    markdown = (
        "# ByT5: Towards a Token-Free Future with Pre-trained\n"
        "Byte-to-Byte Models\n"
        "###### Abstract\n"
    )
    assert _extract_title(markdown) == (
        "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
    )


def test_extract_title_joins_connector_prefixed_continuation() -> None:
    markdown = (
        "T5G2P: Using Text-to-Text Transfer Transformer\n"
        "for Grapheme-to-Phoneme Conversion\n"
        "Abstract\n"
    )
    assert _extract_title(markdown) == (
        "T5G2P: Using Text-to-Text Transfer Transformer "
        "for Grapheme-to-Phoneme Conversion"
    )

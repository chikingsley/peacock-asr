from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from shutil import which
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("peacock_asr.papers")

ARXIV_ID_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")
MIN_ARXIV_HTML_WORDS = 200
MIN_HEADING_COUNT = 3
MAX_WEIRD_RATIO = 0.02
ABSTRACT_PREFACE_MIN_WORDS = 120
TITLE_MIN_LENGTH = 20
TITLE_MAX_LENGTH = 180
TITLE_MIN_WORDS = 4
TITLE_MIN_ALPHA_RATIO = 0.55
TITLE_CONTINUATION_MAX_LENGTH = 90
MAX_NUMERIC_HEADING_TITLE_LENGTH = 90
MAX_NUMERIC_HEADING_TITLE_WORDS = 16
MAX_NUMERIC_HEADING_TITLE_COMMAS = 2
MAX_NUMERIC_HEADING_TITLE_COLON_WORDS = 12
MAX_NUMERIC_HEADING_TITLE_SENTENCE_WORDS = 10
MAX_EXISTING_HEADING_LENGTH = 110
SUSPICIOUS_NUMERIC_HEADING_PREFIXES = (
    "transcript:",
    "ratings:",
)
MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
TITLE_JOIN_TAILS = (
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "pre-trained",
    "pretrained",
    "to",
    "with",
)


@dataclass(slots=True)
class ConvertConfig:
    papers_root: Path
    folder: str | None = None
    force: bool = False
    strict: bool = True
    min_words: int = 700
    timeout_seconds: float = 20.0
    report_path: Path | None = None


@dataclass(slots=True)
class QCResult:
    pass_qc: bool
    warnings: list[str] = field(default_factory=list)
    word_count: int = 0
    heading_count: int = 0
    weird_char_ratio: float = 0.0
    has_abstract: bool = False


@dataclass(slots=True)
class ConvertResult:
    pdf_path: Path
    md_path: Path
    status: str
    extractor: str | None
    arxiv_id: str | None
    source_url: str | None
    title: str | None
    qc: QCResult | None
    error: str | None = None


def convert_papers(config: ConvertConfig) -> list[ConvertResult]:
    pdf_paths = _discover_pdfs(config)
    if not pdf_paths:
        msg = f"No PDF files found under {config.papers_root!s}"
        raise FileNotFoundError(msg)

    results = [
        _convert_single_pdf(pdf_path=pdf_path, config=config)
        for pdf_path in pdf_paths
    ]
    report_path = _resolve_report_path(config)
    _write_report(report_path=report_path, results=results)

    written = sum(1 for result in results if result.status == "written")
    skipped = sum(1 for result in results if result.status == "skipped")
    warnings = sum(1 for result in results if result.status == "warning")
    errors = sum(1 for result in results if result.status == "error")
    logger.info(
        "Paper conversion summary: written=%d skipped=%d warning=%d error=%d report=%s",
        written,
        skipped,
        warnings,
        errors,
        report_path,
    )
    if config.strict and (warnings > 0 or errors > 0):
        msg = (
            "Paper conversion failed strict checks. "
            f"warning={warnings} error={errors} report={report_path}"
        )
        raise RuntimeError(msg)
    return results


def parse_arxiv_id(path: Path) -> str | None:
    match = ARXIV_ID_RE.search(path.stem)
    if match is None:
        return None
    return match.group("id")


def normalize_markdown(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    text = re.sub(r"([A-Za-z])-\n([A-Za-z])", r"\1\2", text)
    text = re.sub(r"(?<=\S)\d{2,}(?=https?://)", " ", text)
    text = re.sub(r"(?<=\S)\d{2,}(?=Previous\b)", "", text)
    text = re.sub(r"(?<=\S)\d{2,}(?=Samples\b)", "", text)
    text = re.sub(r"\b\d{1,3}(?=All the transcriptions\b)", "", text)
    lines = [line.rstrip() for line in text.split("\n")]
    compact = "\n".join(lines)
    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    return compact + "\n"


def promote_section_headings(markdown: str) -> str:
    # Handle canonical paper section headings from PDF text extraction.
    section_patterns = [
        re.compile(
            r"^(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<title>[A-Z][\w\-.,' ,:()/]{2,})$",
        ),
        re.compile(
            r"^(?P<num>[IVXLCM]+)\.\s+(?P<title>[A-Z][\w\- ,:()/]{2,})$",
            flags=re.IGNORECASE,
        ),
    ]
    lexical_sections = {
        "abstract",
        "introduction",
        "related work",
        "method",
        "methods",
        "methodology",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "references",
    }

    def detect_heading(line: str) -> str | None:
        lexical_match = re.match(r"^(?P<section>[A-Za-z ]{3,40})$", line)
        if lexical_match is not None:
            section_name = lexical_match.group("section").strip().lower()
            if section_name in lexical_sections:
                return lexical_match.group("section").strip()

        for pattern in section_patterns:
            match = pattern.match(line)
            if match is not None:
                heading_title = match.group("title").strip()
                if _looks_like_numeric_section_title(heading_title):
                    return f"{match.group('num')} {heading_title}"
        return None

    output_lines: list[str] = []
    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            output_lines.append(raw_line)
            continue
        if stripped.startswith("#"):
            heading_match = re.match(r"^(?P<marks>#{1,6})\s+(?P<text>.+)$", stripped)
            if heading_match is None:
                output_lines.append(raw_line)
                continue

            marks = heading_match.group("marks")
            heading_text = heading_match.group("text").strip()
            if _should_demote_existing_heading(
                heading_text=heading_text,
                level=len(marks),
            ):
                output_lines.append(heading_text)
            else:
                output_lines.append(raw_line)
            continue

        heading = detect_heading(stripped)
        if heading is not None:
            output_lines.append(f"## {heading}")
            continue

        if re.search(r"\s{2,}", stripped):
            first, remainder = re.split(r"\s{2,}", stripped, maxsplit=1)
            first = first.strip()
            remainder = remainder.strip()
            if first and remainder:
                heading = detect_heading(first)
                if heading is not None:
                    output_lines.append(f"## {heading}")
                    output_lines.append(remainder)
                    continue

        output_lines.append(raw_line)

    promoted = "\n".join(output_lines)
    return normalize_markdown(promoted)


def evaluate_markdown_quality(markdown: str, *, min_words: int) -> QCResult:
    word_count = _word_count(markdown)
    heading_count = len(re.findall(r"(?m)^#{1,6}\s+\S", markdown))
    weird_count = len(re.findall(r"[^\x09\x0A\x0D\x20-\x7E]", markdown))
    weird_ratio = weird_count / max(len(markdown), 1)
    has_abstract = _has_abstract_marker_or_preface(markdown)

    warnings: list[str] = []
    if word_count < min_words:
        warnings.append(
            f"word_count={word_count} is below minimum {min_words}",
        )
    if heading_count < MIN_HEADING_COUNT:
        warnings.append(
            "heading_count="
            f"{heading_count} is low (expected >= {MIN_HEADING_COUNT})",
        )
    if not has_abstract:
        warnings.append("missing 'Abstract' section marker")
    if weird_ratio > MAX_WEIRD_RATIO:
        warnings.append(
            "high non-ascii ratio "
            f"({weird_ratio:.3f}; expected <= {MAX_WEIRD_RATIO:.3f})",
        )

    return QCResult(
        pass_qc=not warnings,
        warnings=warnings,
        word_count=word_count,
        heading_count=heading_count,
        weird_char_ratio=weird_ratio,
        has_abstract=has_abstract,
    )


def _discover_pdfs(config: ConvertConfig) -> list[Path]:
    root = config.papers_root
    if config.folder:
        root = root / config.folder
    if not root.exists():
        msg = f"Path does not exist: {root}"
        raise FileNotFoundError(msg)
    return sorted(path for path in root.rglob("*.pdf") if path.is_file())


def _convert_single_pdf(*, pdf_path: Path, config: ConvertConfig) -> ConvertResult:
    md_path = pdf_path.with_suffix(".md")
    if md_path.exists() and not config.force:
        return ConvertResult(
            pdf_path=pdf_path,
            md_path=md_path,
            status="skipped",
            extractor=None,
            arxiv_id=parse_arxiv_id(pdf_path),
            source_url=None,
            title=None,
            qc=None,
        )

    arxiv_id = parse_arxiv_id(pdf_path)
    extractor: str | None = None
    source_url: str | None = None
    markdown: str | None = None

    if arxiv_id is not None:
        fetched = _fetch_arxiv_html(arxiv_id=arxiv_id, timeout=config.timeout_seconds)
        if fetched is not None:
            html, source_url = fetched
            candidate_md = _extract_markdown_from_html(html)
            if (
                candidate_md
                and _word_count(normalize_markdown(candidate_md))
                >= MIN_ARXIV_HTML_WORDS
            ):
                markdown = candidate_md
                extractor = "trafilatura-arxiv-html"

    if markdown:
        final_md, title = _prepare_markdown(markdown)
        qc_result = evaluate_markdown_quality(final_md, min_words=config.min_words)
        status = "written" if qc_result.pass_qc else "warning"

        if status == "warning":
            try:
                fallback_md, fallback_title, fallback_qc, fallback_extractor = (
                    _extract_best_pdf_markdown(
                        pdf_path=pdf_path,
                        min_words=config.min_words,
                    )
                )
            except RuntimeError:
                fallback_md = None
                fallback_title = None
                fallback_qc = None
                fallback_extractor = None
            if (
                fallback_md is not None
                and fallback_qc is not None
                and _is_better_qc(candidate=fallback_qc, baseline=qc_result)
            ):
                final_md = fallback_md
                title = fallback_title
                qc_result = fallback_qc
                status = "written" if fallback_qc.pass_qc else "warning"
                extractor = fallback_extractor
    else:
        try:
            final_md, title, qc_result, extractor = _extract_best_pdf_markdown(
                pdf_path=pdf_path,
                min_words=config.min_words,
            )
        except RuntimeError as exc:
            return ConvertResult(
                pdf_path=pdf_path,
                md_path=md_path,
                status="error",
                extractor=None,
                arxiv_id=arxiv_id,
                source_url=source_url,
                title=None,
                qc=None,
                error=str(exc),
            )
        status = "written" if qc_result.pass_qc else "warning"

    md_path.write_text(final_md, encoding="utf-8")
    return ConvertResult(
        pdf_path=pdf_path,
        md_path=md_path,
        status=status,
        extractor=extractor,
        arxiv_id=arxiv_id,
        source_url=source_url,
        title=title,
        qc=qc_result,
    )


def _word_count(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", text)
    return len(words)


def _looks_like_numeric_section_title(title: str) -> bool:
    lowered = title.lower()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", title)
    word_count = len(words)
    if (
        len(title) > MAX_NUMERIC_HEADING_TITLE_LENGTH
        or word_count > MAX_NUMERIC_HEADING_TITLE_WORDS
        or title.count(",") > MAX_NUMERIC_HEADING_TITLE_COMMAS
    ):
        return False
    starts_with_suspicious_prefix = any(
        lowered.startswith(prefix)
        for prefix in SUSPICIOUS_NUMERIC_HEADING_PREFIXES
    )
    if starts_with_suspicious_prefix:
        return False
    if ". " in title and word_count > MAX_NUMERIC_HEADING_TITLE_SENTENCE_WORDS:
        return False
    return not (
        ":" in title and word_count > MAX_NUMERIC_HEADING_TITLE_COLON_WORDS
    )


def _should_demote_existing_heading(*, heading_text: str, level: int) -> bool:
    if level <= 1:
        return False

    lowered = heading_text.lower()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", heading_text)
    long_heading = (
        len(heading_text) > MAX_EXISTING_HEADING_LENGTH
        and len(words) > MAX_NUMERIC_HEADING_TITLE_WORDS
    )
    is_verbose_transcript = (
        lowered.startswith("transcript:")
        and len(heading_text) > MAX_NUMERIC_HEADING_TITLE_LENGTH
    )
    return long_heading or is_verbose_transcript


def _extract_best_pdf_markdown(
    *,
    pdf_path: Path,
    min_words: int,
) -> tuple[str, str | None, QCResult, str]:
    candidates: list[tuple[str, str | None, QCResult, str, float]] = []
    errors: list[str] = []
    attempts = (
        (False, "pdftotext-layout"),
        (True, "pdftotext-raw"),
    )
    for use_raw, label in attempts:
        try:
            pdf_text = _extract_markdown_from_pdf(pdf_path, use_raw=use_raw)
        except RuntimeError as exc:
            errors.append(f"{label}: {exc}")
            continue

        final_md, title = _prepare_markdown(pdf_text)
        qc_result = evaluate_markdown_quality(final_md, min_words=min_words)
        artifact_ratio = _column_artifact_ratio(final_md)
        candidates.append((final_md, title, qc_result, label, artifact_ratio))

    if not candidates:
        msg = (
            " and ".join(errors)
            if errors
            else f"pdftotext failed for {pdf_path.name}"
        )
        raise RuntimeError(msg)

    def candidate_key(item: tuple[str, str | None, QCResult, str, float]) -> tuple:
        qc = item[2]
        artifact_ratio = item[4]
        return (
            len(qc.warnings),
            artifact_ratio,
            -qc.heading_count,
            -qc.word_count,
        )

    best = min(candidates, key=candidate_key)
    return best[0], best[1], best[2], best[3]


def _column_artifact_ratio(markdown: str) -> float:
    lines = [line for line in markdown.splitlines() if line.strip()]
    if not lines:
        return 0.0
    affected = sum(1 for line in lines if re.search(r"\S\s{12,}\S", line))
    return affected / len(lines)


def _has_abstract_marker_or_preface(markdown: str) -> bool:
    lowered = markdown.lower()
    if re.search(r"(?mi)^\s*#{0,3}\s*abstract\b", lowered):
        return True

    intro_match = re.search(
        r"(?mi)^\s*#{1,3}\s*(?:\d+(?:\.\d+)*\.?\s+)?introduction\b",
        markdown,
    )
    if intro_match is None:
        return False

    preface = markdown[: intro_match.start()]
    return _word_count(preface) >= ABSTRACT_PREFACE_MIN_WORDS


def _prepare_markdown(markdown: str) -> tuple[str, str | None]:
    normalized = normalize_markdown(markdown)
    title = _extract_title(normalized)
    final_md = _ensure_title_heading(markdown=normalized, title=title)
    final_md = promote_section_headings(final_md)
    return final_md, title


def _is_better_qc(*, candidate: QCResult, baseline: QCResult) -> bool:
    candidate_warning_count = len(candidate.warnings)
    baseline_warning_count = len(baseline.warnings)
    if candidate_warning_count < baseline_warning_count:
        return True
    if candidate_warning_count > baseline_warning_count:
        return False
    if candidate.word_count > baseline.word_count:
        return True
    if candidate.word_count < baseline.word_count:
        return False
    return candidate.heading_count > baseline.heading_count


def _fetch_arxiv_html(
    *,
    arxiv_id: str,
    timeout: float,
) -> tuple[str, str] | None:
    candidates = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.org/html/{arxiv_id}",
    ]
    for url in candidates:
        request = Request(  # noqa: S310
            url=url,
            headers={"User-Agent": "peacock-asr/0.1 papers-convert"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                charset = response.headers.get_content_charset() or "utf-8"
                payload = response.read().decode(charset, errors="replace")
                return payload, url
        except (HTTPError, URLError, TimeoutError, ValueError, OSError):
            continue
    return None


def _extract_markdown_from_html(html: str) -> str | None:
    try:
        import trafilatura  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            "trafilatura is required for arXiv HTML extraction. "
            "Install dependency: uv add trafilatura"
        )
        raise RuntimeError(msg) from exc

    return trafilatura.extract(  # type: ignore[no-any-return]
        html,
        output_format="markdown",
        favor_precision=True,
        include_links=False,
    )


def _extract_markdown_from_pdf(pdf_path: Path, *, use_raw: bool = False) -> str:
    pdftotext_bin = which("pdftotext")
    if pdftotext_bin is None:
        msg = (
            "pdftotext is required for fallback extraction but was not found in PATH"
        )
        raise RuntimeError(msg)

    command = [pdftotext_bin]
    if use_raw:
        command.append("-raw")
    else:
        command.append("-layout")
    command.extend((str(pdf_path), "-"))

    completed = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        msg = f"pdftotext failed for {pdf_path.name}: {stderr}"
        raise RuntimeError(msg)
    content = completed.stdout.strip()
    if not content:
        msg = f"pdftotext returned no content for {pdf_path.name}"
        raise RuntimeError(msg)
    return content


def _extract_title(markdown: str) -> str | None:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    limit = min(80, len(lines))

    for index, line in enumerate(lines[:limit]):
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if not heading:
                continue
            continuation = lines[index + 1] if index + 1 < len(lines) else None
            return _merge_wrapped_title(
                title=heading,
                next_line=continuation,
            )[:200]

    for index, line in enumerate(lines[:limit]):
        if _looks_like_title_candidate(line):
            continuation = lines[index + 1] if index + 1 < len(lines) else None
            return _merge_wrapped_title(
                title=line,
                next_line=continuation,
            )[:200]

    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return None


def _looks_like_title_candidate(line: str) -> bool:
    if len(line) < TITLE_MIN_LENGTH or len(line) > TITLE_MAX_LENGTH:
        return False
    lowered = line.lower()
    starts_with_noise = lowered.startswith(
        ("arxiv:", "https://", "http://", "copyright"),
    )
    has_month = any(month in lowered for month in MONTH_NAMES)
    looks_like_date = has_month and bool(re.search(r"\b\d{4}\b", lowered))
    looks_like_conference = bool(
        re.match(r"(?i)^(inter|intra)?speech\s+\d{4}", line)
        or re.match(r"(?i)^(icassp|neurips|interspeech|acl|emnlp)\b", line),
    )
    has_alpha = bool(re.search(r"[A-Za-z]", line))
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", line)
    alpha_chars = sum(1 for char in line if char.isalpha())
    alpha_ratio = alpha_chars / max(len(line), 1)
    invalid = (
        starts_with_noise
        or looks_like_date
        or looks_like_conference
        or not has_alpha
        or len(words) < TITLE_MIN_WORDS
        or line.endswith((".", ";", ":"))
        or alpha_ratio < TITLE_MIN_ALPHA_RATIO
    )
    return not invalid


def _merge_wrapped_title(*, title: str, next_line: str | None) -> str:
    candidate = (next_line or "").strip()
    candidate_word_count = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'.-]*", candidate))

    title_tail = title.rstrip().rsplit(" ", 1)[-1].lower()
    title_ends_with_connector = title_tail in TITLE_JOIN_TAILS
    title_ends_with_hyphen = title.rstrip().endswith("-")
    candidate_head = candidate.split(" ", 1)[0].lower()
    candidate_starts_with_connector = candidate_head in TITLE_JOIN_TAILS
    blocked_candidate = (
        not candidate
        or candidate.lower() in {"abstract", "introduction", "index terms"}
        or candidate.startswith(("#", "##"))
        or len(candidate) > TITLE_CONTINUATION_MAX_LENGTH
    )
    should_join = (
        not blocked_candidate
        and (
            title_ends_with_connector
            or title_ends_with_hyphen
            or candidate_starts_with_connector
        )
        and candidate_word_count >= 2  # noqa: PLR2004
    )
    return f"{title} {candidate}" if should_join else title


def _ensure_title_heading(*, markdown: str, title: str | None) -> str:
    if title is None:
        return markdown

    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            lines[index] = f"# {title}"

            for next_index in range(index + 1, len(lines)):
                next_line = lines[next_index].strip()
                if not next_line:
                    continue
                if (
                    not next_line.startswith("#")
                    and title.endswith(next_line)
                ):
                    lines.pop(next_index)
                break
            return _dedupe_leading_title_line("\n".join(lines), title=title)

        prefixed = f"# {title}\n\n{markdown.lstrip()}"
        return _dedupe_leading_title_line(prefixed, title=title)
    return markdown


def _dedupe_leading_title_line(markdown: str, *, title: str) -> str:
    lines = markdown.splitlines()

    heading_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip().startswith("#"):
            heading_index = index
            break
    if heading_index is None:
        return normalize_markdown(markdown)

    body_indices: list[int] = []
    for next_index in range(heading_index + 1, len(lines)):
        next_line = lines[next_index].strip()
        if not next_line:
            continue
        if next_line.startswith("#"):
            break
        body_indices.append(next_index)
        if len(body_indices) == 2:  # noqa: PLR2004
            break

    if body_indices:
        first_line = lines[body_indices[0]].strip()
        if title.endswith(first_line):
            lines.pop(body_indices[0])
        elif len(body_indices) == 2:  # noqa: PLR2004
            second_line = lines[body_indices[1]].strip()
            combined = f"{first_line} {second_line}"
            if combined == title:
                lines.pop(body_indices[1])
                lines.pop(body_indices[0])

    return normalize_markdown("\n".join(lines))


def _resolve_report_path(config: ConvertConfig) -> Path:
    if config.report_path is not None:
        return config.report_path
    root = config.papers_root / config.folder if config.folder else config.papers_root
    return root / "conversion_report.json"


def _write_report(*, report_path: Path, results: list[ConvertResult]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "items": [_result_to_dict(result=result) for result in results],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _result_to_dict(*, result: ConvertResult) -> dict[str, Any]:
    return {
        "pdf_path": str(result.pdf_path),
        "md_path": str(result.md_path),
        "status": result.status,
        "extractor": result.extractor,
        "arxiv_id": result.arxiv_id,
        "source_url": result.source_url,
        "title": result.title,
        "error": result.error,
        "qc": _qc_to_dict(qc=result.qc),
    }


def _qc_to_dict(*, qc: QCResult | None) -> dict[str, Any] | None:
    if qc is None:
        return None
    return {
        "pass_qc": qc.pass_qc,
        "warnings": qc.warnings,
        "word_count": qc.word_count,
        "heading_count": qc.heading_count,
        "weird_char_ratio": qc.weird_char_ratio,
        "has_abstract": qc.has_abstract,
    }

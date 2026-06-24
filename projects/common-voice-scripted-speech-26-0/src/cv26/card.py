"""Generate the Hugging Face dataset card (README + LICENSE) from the manifest and pipeline state.

The card describes the parquet deliverable and lists every manifest language with its current
upload status, derived from :func:`cv26.pipeline.load_state`. It regenerates deterministically, so
``cv26 card --upload`` can refresh it whenever languages are added or uploaded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cv26 import config
from cv26.manifest import load_manifest
from cv26.pipeline import load_state

if TYPE_CHECKING:
    from pathlib import Path

    from cv26.manifest import ManifestRow

LOGGER = logging.getLogger("cv26.card")

LICENSE_TEXT = """SPDX-License-Identifier: CC0-1.0

Common Voice Scripted Speech is reported by Mozilla Data Collective as Creative Commons
Zero v1.0 Universal.

Upstream source: https://mozilladatacollective.com/
SPDX license record: https://spdx.org/licenses/CC0-1.0.html
Creative Commons deed: https://creativecommons.org/publicdomain/zero/1.0/

This repository preserves upstream source dataset IDs, locales, and archive filenames so
downstream users can trace every normalized row back to its Mozilla Data Collective source.
"""


def _language_yaml(rows: list[ManifestRow]) -> list[str]:
    locales = sorted({row.locale for row in rows if row.locale})
    plain = [loc for loc in locales if "-" not in loc]
    bcp47 = [loc for loc in locales if "-" in loc]
    lines = ["language:", *[f"- {loc}" for loc in plain]]
    if bcp47:
        lines += ["language_bcp47:", *[f"- {tag}" for tag in bcp47]]
    return lines


def render_card(rows: list[ManifestRow], status: dict[str, str]) -> str:
    """Render the dataset README markdown from manifest rows and upload status."""
    uploaded = sum(1 for row in rows if status.get(row.dataset_id) == "complete")
    table = [
        "| Language | Locale | MDC dataset ID | Status |",
        "| --- | --- | --- | --- |",
        *(
            f"| {row.language} | `{row.locale}` | `{row.dataset_id}` | "
            f"{status.get(row.dataset_id, 'pending')} |"
            for row in sorted(rows, key=lambda r: r.language)
        ),
    ]
    return "\n".join(
        [
            "---",
            "license: cc0-1.0",
            *_language_yaml(rows),
            "task_categories:",
            "- automatic-speech-recognition",
            "tags:",
            "- common-voice",
            "- mozilla-data-collective",
            "- scripted-speech",
            "- multilingual-asr",
            "- asr",
            "pretty_name: Common Voice Scripted Speech",
            "---",
            "",
            "# Common Voice Scripted Speech",
            "",
            "A row-normalized multilingual ASR dataset built from Mozilla Data Collective",
            "Common Voice Scripted Speech. Each upstream archive is converted to appendable",
            "parquet shards under `data/<upstream_split>/`, one shard per source archive and",
            "split, with audio bytes embedded in an `audio` struct column.",
            "",
            "## Status",
            "",
            f"- Manifest languages: {len(rows)}",
            f"- Languages uploaded: {uploaded}",
            "",
            "## Columns",
            "",
            "- `audio` (`bytes`, `path`)",
            "- `sentence`, `locale`, `language`, `upstream_split`",
            "- `source_dataset_id`, `source_archive`, `collection`",
            "- upstream Common Voice metadata: `client_id`, `sentence_id`, `sentence_domain`,",
            "  `up_votes`, `down_votes`, `age`, `gender`, `accents`, `variant`, `segment`",
            "- `duration_ms` from `clip_durations.tsv` when available",
            "- `license`, `license_url`",
            "",
            "## License",
            "",
            "Mozilla Data Collective records these archives as Creative Commons Zero v1.0",
            "Universal (`CC0-1.0`). See `LICENSE` and <https://spdx.org/licenses/CC0-1.0.html>.",
            "",
            "## Languages",
            "",
            *table,
            "",
        ],
    )


def write_card() -> tuple[Path, Path]:
    """Write README.md and LICENSE next to the project; return their paths."""
    rows = load_manifest(config.MANIFEST)
    status = load_state(config.PROCESSING_STATE)
    readme = config.PROJECT / "HF_README.md"
    license_path = config.PROJECT / "data" / "hf-parquet" / "LICENSE"
    readme.write_text(render_card(rows, status), encoding="utf-8")
    license_path.parent.mkdir(parents=True, exist_ok=True)
    license_path.write_text(LICENSE_TEXT, encoding="utf-8")
    LOGGER.info("wrote card: %s", readme)
    return readme, license_path

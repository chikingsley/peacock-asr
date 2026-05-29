from __future__ import annotations

import argparse
import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from p016_compare.pipeline import PronunciationComparePipeline

PROMPTS = {
    "ru_1": "Сегодня хорошая погода, я иду в магазин.",
    "ru_2": "Я хочу говорить по-русски ясно и спокойно.",
    "en_1": "The quick brown fox jumps over the lazy dog.",
    "en_2": "I would like a glass of water, please.",
}
WORD_COLUMNS = [
    "word",
    "target phones",
    "recognized phones",
    "PER",
    "PFER",
    "subs",
    "dels",
    "ins",
    "substitutions",
    "deletions",
    "insertions",
]


@lru_cache(maxsize=1)
def _pipeline() -> PronunciationComparePipeline:
    return PronunciationComparePipeline()


def analyze(
    audio_path: str | None,
    language: str,
    prompt: str,
) -> tuple[str, str, str, str, str, list[list[Any]], list[list[Any]], str]:
    del prompt
    if not audio_path:
        return "No audio.", "", "", "", "", [], [], ""

    result = _pipeline().analyze(
        Path(audio_path),
        _language_code(language),
    )
    payload = result.as_dict()
    lanes = {lane["name"]: lane for lane in payload["lanes"]}
    zipa = lanes.get("zipa", {})
    xlsr = lanes.get("xlsr-espeak", {})

    zipa_html = _word_highlights(zipa)
    xlsr_html = _word_highlights(xlsr)
    zipa_text = json.dumps(_lane_summary(zipa), ensure_ascii=False, indent=2)
    xlsr_text = json.dumps(_lane_summary(xlsr), ensure_ascii=False, indent=2)
    target_text = json.dumps(payload["targets"], ensure_ascii=False, indent=2)
    asr_text = json.dumps(payload["asr"], ensure_ascii=False, indent=2)
    return (
        asr_text,
        zipa_html,
        xlsr_html,
        zipa_text,
        xlsr_text,
        _word_table(zipa),
        _word_table(xlsr),
        target_text,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="P016 Phone Compare") as demo:
        with gr.Row():
            prompt = gr.Textbox(value=PROMPTS["ru_1"], label="Practice text")
            language = gr.Dropdown(
                choices=["Russian", "English"],
                value="Russian",
                label="Language",
            )
        with gr.Row():
            for label, key in [
                ("RU 1", "ru_1"),
                ("RU 2", "ru_2"),
                ("EN 1", "en_1"),
                ("EN 2", "en_2"),
            ]:
                button = gr.Button(label)
                selection = (PROMPTS[key], _prompt_language(key))
                button.click(
                    lambda selected=selection: selected,
                    outputs=[prompt, language],
                )

        audio = gr.Audio(
            sources="microphone",
            type="filepath",
            label="Microphone",
            format="wav",
            editable=False,
        )
        run = gr.Button("Analyze last recording", variant="primary")

        with gr.Row():
            zipa_highlights = gr.HTML()
            xlsr_highlights = gr.HTML()
        with gr.Row():
            zipa_words = gr.Dataframe(headers=WORD_COLUMNS, label="ZIPA words")
            xlsr_words = gr.Dataframe(headers=WORD_COLUMNS, label="XLSR-eSpeak words")
        with gr.Accordion("Debug", open=False):
            asr_out = gr.Code(label="Qwen ASR hypothesis", language="json")
            with gr.Row():
                zipa_out = gr.Code(label="ZIPA", language="json")
                xlsr_out = gr.Code(label="XLSR-eSpeak", language="json")
            target_out = gr.Code(label="ASR hypothesis -> target G2P", language="json")

        run.click(
            analyze,
            inputs=[audio, language, prompt],
            outputs=[
                asr_out,
                zipa_highlights,
                xlsr_highlights,
                zipa_out,
                xlsr_out,
                zipa_words,
                xlsr_words,
                target_out,
            ],
        )
        audio.stop_recording(
            analyze,
            inputs=[audio, language, prompt],
            outputs=[
                asr_out,
                zipa_highlights,
                xlsr_highlights,
                zipa_out,
                xlsr_out,
                zipa_words,
                xlsr_words,
                target_out,
            ],
            trigger_mode="always_last",
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name=args.server_name, server_port=args.server_port)


def _language_code(language: str) -> str:
    if language == "English":
        return "en_us"
    return "ru"


def _prompt_language(key: str) -> str:
    return "English" if key.startswith("en_") else "Russian"


def _lane_summary(lane: dict[str, Any]) -> dict[str, Any]:
    if not lane:
        return {}
    return {
        "model_id": lane.get("model_id"),
        "error": lane.get("error"),
        "target": {
            "backend": lane.get("target", {}).get("backend"),
            "warnings": lane.get("target", {}).get("warnings"),
        },
        "sentence": lane.get("sentence"),
        "raw_text": lane.get("raw_text"),
        "raw_tokens": lane.get("raw_tokens"),
    }


def _word_highlights(lane: dict[str, Any]) -> str:
    title = html.escape(str(lane.get("name") or "model"))
    error = lane.get("error")
    if error:
        escaped_error = html.escape(str(error))
        return (
            f"<section class='p016-panel'><h3>{title}</h3>"
            f"<p class='p016-error'>{escaped_error}</p></section>"
        )

    words = list(lane.get("words", []))
    sentence = lane.get("sentence", {})
    badges = (
        f"PER {_percent(sentence.get('PER', 0))} "
        f"· PFER {_percent(sentence.get('PFER', 0))} "
        f"· S {sentence.get('substitutions', 0)} "
        f"D {sentence.get('deletions', 0)} "
        f"I {sentence.get('insertions', 0)}"
    )
    chips = []
    for row in words:
        chips.append(_word_chip(row))
    return (
        "<style>"
        ".p016-panel{background:#202124;border:1px solid #383a40;"
        "border-radius:8px;padding:12px;min-height:130px}"
        ".p016-panel h3{margin:0 0 8px 0;font-size:14px;color:#f4f4f5}"
        ".p016-badges{font-size:12px;color:#c8c8cc;margin-bottom:10px}"
        ".p016-words{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-start}"
        ".p016-word{border:1px solid #4b5563;border-radius:8px;"
        "padding:7px 9px;min-width:64px;background:#2b2d31}"
        ".p016-ok{border-color:#3c8f52;background:#173320}"
        ".p016-warn{border-color:#d99a24;background:#3a2b12}"
        ".p016-bad{border-color:#d04c4c;background:#3a1717}"
        ".p016-token{font-weight:700;color:#ffffff;font-size:14px}"
        ".p016-mini{font-size:11px;color:#d6d6d8;margin-top:3px;white-space:nowrap}"
        ".p016-detail{font-size:11px;color:#b8b8bd;margin-top:3px;max-width:180px;overflow:hidden;text-overflow:ellipsis}"
        ".p016-error{color:#ffb4b4;font-size:12px;white-space:pre-wrap}"
        "</style>"
        f"<section class='p016-panel'><h3>{title}</h3>"
        f"<div class='p016-badges'>{html.escape(badges)}</div>"
        f"<div class='p016-words'>{''.join(chips)}</div></section>"
    )


def _word_chip(row: dict[str, Any]) -> str:
    per = float(row.get("PER") or 0.0)
    pfer = float(row.get("PFER") or 0.0)
    css = "p016-ok" if per == 0 and pfer == 0 else "p016-warn" if per <= 0.34 else "p016-bad"
    details = []
    if row.get("substitutions_detail"):
        details.append(f"sub {row['substitutions_detail']}")
    if row.get("deletions_detail"):
        details.append(f"del {row['deletions_detail']}")
    if row.get("insertions_detail"):
        details.append(f"ins {row['insertions_detail']}")
    detail = "; ".join(str(item) for item in details) or "match"
    title = (
        f"target: {row.get('target_phones', '')}\n"
        f"recognized: {row.get('recognized_phones', '')}\n"
        f"{detail}"
    )
    return (
        f"<span class='p016-word {css}' title='{html.escape(title)}'>"
        f"<div class='p016-token'>{html.escape(str(row.get('word', '')))}</div>"
        f"<div class='p016-mini'>PER {_percent(per)} · PFER {_percent(pfer)}</div>"
        f"<div class='p016-detail'>{html.escape(detail)}</div>"
        "</span>"
    )


def _word_table(lane: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for row in lane.get("words", []):
        rows.append(
            [
                row.get("word", ""),
                row.get("target_phones", ""),
                row.get("recognized_phones", ""),
                _percent(row.get("PER", 0)),
                _percent(row.get("PFER", 0)),
                row.get("substitutions", 0),
                row.get("deletions", 0),
                row.get("insertions", 0),
                row.get("substitutions_detail", ""),
                row.get("deletions_detail", ""),
                row.get("insertions_detail", ""),
            ]
        )
    return rows


def _percent(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"

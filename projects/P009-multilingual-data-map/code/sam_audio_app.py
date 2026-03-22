#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gradio>=5.24.0",
#     "torch>=2.0",
#     "torchaudio>=2.0",
#     "python-dotenv>=1.0",
#     "setuptools<70",
#     "sam-audio @ git+https://github.com/facebookresearch/sam-audio.git",
# ]
# ///
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from sam_audio_test import run_single_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "ru_open_stt" / "sam_test_out"


def separate_audio(
    audio_path: str | None,
    description: str,
    use_judge: bool,
):
    if not audio_path:
        raise gr.Error("Upload an audio file first.")
    prompt = description.strip()
    if not prompt:
        raise gr.Error("Enter a text prompt like 'a person speaking'.")

    result = run_single_file(
        audio_path=Path(audio_path),
        description=prompt,
        out_dir=DEFAULT_OUTPUT_DIR,
        with_judge=use_judge,
    )
    lines = [
        f"file: {result['file']}",
        f"duration: {result['dur']:.1f}s",
    ]
    if result["score"] is not None:
        lines.append(f"judge score: {result['score']:.3f}")
    lines.append(f"original: {result['original_path']}")
    lines.append(f"target: {result['target_path']}")
    return (
        str(result["original_path"]),
        str(result["target_path"]),
        "\n".join(lines),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SAM-Audio Demo") as demo:
        gr.Markdown(
            "# SAM-Audio Demo\n"
            "Upload a clip, describe the sound you want, and compare the original with the separated target."
        )
        with gr.Row():
            audio_in = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Input audio",
            )
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a person speaking",
                )
                use_judge = gr.Checkbox(
                    label="Run judge model",
                    value=False,
                )
                run_btn = gr.Button("Separate", variant="primary")
        with gr.Row():
            original_out = gr.Audio(label="Original", type="filepath")
            target_out = gr.Audio(label="Separated target", type="filepath")
        status = gr.Textbox(label="Status", lines=6)

        run_btn.click(
            fn=separate_audio,
            inputs=[audio_in, prompt, use_judge],
            outputs=[original_out, target_out, status],
        )
    return demo


if __name__ == "__main__":
    host = os.environ.get("SAM_AUDIO_APP_HOST", "0.0.0.0")
    port = int(os.environ.get("SAM_AUDIO_APP_PORT", "7860"))
    build_app().launch(server_name=host, server_port=port, show_error=True)

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from p016_compare.normalization import split_phone_text


@dataclass(frozen=True)
class PhoneRecognitionResult:
    name: str
    model_id: str
    raw_text: str
    raw_tokens: list[str]
    normalized_tokens: list[str]
    error: str | None = None

    @classmethod
    def failed(cls, name: str, model_id: str, error: Exception | str) -> PhoneRecognitionResult:
        return cls(
            name=name,
            model_id=model_id,
            raw_text="",
            raw_tokens=[],
            normalized_tokens=[],
            error=str(error),
        )


class ZipaOnnxRecognizer:
    name = "zipa"
    model_id = "anyspeech/zipa-large-crctc-ns-800k"

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        tokens_path: str | Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        self.repo_dir = Path(repo_dir or os.getenv("ZIPA_REPO", root / "third_party" / "zipa"))
        self.model_path = Path(model_path or os.getenv("ZIPA_ONNX", _default_zipa_model(root)))
        self.tokens_path = Path(
            tokens_path or os.getenv("ZIPA_TOKENS", self.model_path.parent / "tokens.txt")
        )

    def recognize(self, audio_path: str | Path) -> PhoneRecognitionResult:
        resolved_audio_path = Path(audio_path).resolve()
        script = self.repo_dir / "inference" / "inference.py"
        if not script.exists():
            raise RuntimeError(f"ZIPA inference script not found: {script}")
        if not self.model_path.exists():
            raise RuntimeError(f"ZIPA ONNX model not found: {self.model_path}")
        if not self.tokens_path.exists():
            raise RuntimeError(f"ZIPA tokens file not found: {self.tokens_path}")

        proc = subprocess.run(
            [
                sys.executable,
                str(script),
                str(resolved_audio_path),
                "--model-path",
                str(self.model_path),
                "--model-type",
                "ctc",
                "--tokens",
                str(self.tokens_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            cwd=str(self.repo_dir),
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

        raw = _last_nonempty_line(proc.stdout)
        tokens = split_phone_text(raw)
        return PhoneRecognitionResult(
            name=self.name,
            model_id=self.model_id,
            raw_text=raw,
            raw_tokens=raw.split(),
            normalized_tokens=tokens,
        )


def safe_recognize(
    recognizer: ZipaOnnxRecognizer,
    audio_path: str | Path,
) -> PhoneRecognitionResult:
    try:
        return recognizer.recognize(audio_path)
    except Exception as exc:  # noqa: BLE001 - intentional: surface any recognizer failure as a failed result
        return PhoneRecognitionResult.failed(recognizer.name, recognizer.model_id, exc)


def _default_zipa_model(root: Path) -> str:
    artifact_dir = root / "artifacts" / "zipa-large-crctc-ns-800k"
    fp32 = artifact_dir / "model.onnx"
    if fp32.exists():
        return str(fp32)
    return str(artifact_dir / "model.fp16.onnx")


def _last_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""

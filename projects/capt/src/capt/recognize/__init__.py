"""Phone-recognition lane: audio -> produced IPA (ZIPA universal recognizer, in-process ONNX)."""

from __future__ import annotations

from capt.recognize.zipa import PhoneRecognitionResult, ZipaOnnxRecognizer, safe_recognize

__all__ = ["PhoneRecognitionResult", "ZipaOnnxRecognizer", "safe_recognize"]

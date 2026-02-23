from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ZIPA uses 127 IPA tokens. The mapping is similar to xlsr-espeak but
# the exact vocab comes from the model config. We load it dynamically.
ARPABET_TO_IPA: dict[str, str] = {
    "AA": "\u0251",
    "AE": "\u00e6",
    "AH": "\u0259",
    "AO": "\u0254",
    "AW": "a\u028a",
    "AY": "a\u026a",
    "B": "b",
    "CH": "t\u0283",
    "D": "d",
    "DH": "\u00f0",
    "EH": "\u025b",
    "ER": "\u025d",
    "EY": "e\u026a",
    "F": "f",
    "G": "\u0261",
    "HH": "h",
    "IH": "\u026a",
    "IY": "i",
    "JH": "d\u0292",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "\u014b",
    "OW": "o\u028a",
    "OY": "\u0254\u026a",
    "P": "p",
    "R": "\u0279",
    "S": "s",
    "SH": "\u0283",
    "T": "t",
    "TH": "\u03b8",
    "UH": "\u028a",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "\u0292",
}


class ZIPABackend:
    """ZIPA-CR via ONNX Runtime: 127 IPA tokens, 20ms frames."""

    def __init__(self) -> None:
        self._session: object | None = None
        self._vocab_list: list[str] = []
        self._token_to_idx: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "ZIPA-CR (ONNX)"

    @property
    def vocab(self) -> list[str]:
        return self._vocab_list

    @property
    def blank_index(self) -> int:
        return self._token_to_idx.get("<blk>", 0)

    def load(self) -> None:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as e:
            msg = "onnxruntime is required for the ZIPA backend: uv add onnxruntime"
            raise ImportError(msg) from e

        from gopt_bench.settings import settings  # noqa: PLC0415

        model_dir = settings.models_dir
        model_path = model_dir / "zipa-large-crctc-ns-800k.onnx"
        vocab_path = model_dir / "zipa-vocab.json"

        if not model_path.exists():
            msg = (
                f"ZIPA ONNX model not found at {model_path}. "
                "Download from HuggingFace anyspeech/zipa-large"
                "-crctc-ns-800k and place the .onnx file in "
                "the models directory."
            )
            raise FileNotFoundError(msg)

        logger.info("Loading ZIPA ONNX model from %s", model_path)
        self._session = ort.InferenceSession(str(model_path))

        if vocab_path.exists():
            import json  # noqa: PLC0415

            with vocab_path.open() as f:
                vocab_data = json.load(f)
            self._vocab_list = (
                vocab_data if isinstance(vocab_data, list) else list(vocab_data.keys())
            )
            self._token_to_idx = {t: i for i, t in enumerate(self._vocab_list)}
        else:
            logger.warning(
                "No vocab file at %s, phone mapping unavailable",
                vocab_path,
            )

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch  # noqa: PLC0415
        from lhotse.features.kaldi.extractors import Fbank, FbankConfig  # noqa: PLC0415

        if self._session is None:
            msg = "Call load() before get_posteriors()"
            raise RuntimeError(msg)

        expected_sr = 16000
        if sample_rate != expected_sr:
            msg = f"ZIPA expects 16kHz audio, got {sample_rate}Hz"
            raise ValueError(msg)

        # Extract 80-dim FBank features (ZIPA input format)
        extractor = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        features = extractor.extract_batch([audio_tensor], sampling_rate=expected_sr)
        feature = features[0].unsqueeze(0).numpy()  # [1, T, 80]
        feat_lens = np.array([feature.shape[1]], dtype=np.int64)

        outputs = self._session.run(None, {"x": feature, "x_lens": feat_lens})
        log_probs = outputs[0][0]  # [T, V]

        # Convert log-probs to posteriors
        max_lp = log_probs.max(axis=-1, keepdims=True)
        posteriors = np.exp(log_probs - max_lp)
        posteriors = posteriors / posteriors.sum(axis=-1, keepdims=True)

        return posteriors.astype(np.float64)

    def map_phone(self, arpabet_phone: str) -> int | None:
        ipa = ARPABET_TO_IPA.get(arpabet_phone)
        if ipa is None:
            return None
        return self._token_to_idx.get(ipa)

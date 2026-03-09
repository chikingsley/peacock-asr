"""OmniASR CTC backend via a persistent Python 3.12 fairseq2 worker."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from p003_compact.omni_assets import load_ordered_vocab
from p003_compact.settings import settings

logger = logging.getLogger(__name__)
EXPECTED_SAMPLE_RATE = 16_000

if TYPE_CHECKING:
    from collections.abc import Sequence


class OmniCTCBackend:
    """Loads a fine-tuned OmniASR phoneme run through a persistent helper."""

    def __init__(self, run_ref: str) -> None:
        self._run_ref = run_ref
        self._process: subprocess.Popen[str] | None = None
        self._request_root = settings.cache_dir / "omni_worker"
        self._vocab = load_ordered_vocab()
        self._blank_index = self._vocab.index("[PAD]")
        self._phone_to_idx = {
            token: idx
            for idx, token in enumerate(self._vocab)
            if token not in {"[UNK]", "[PAD]"}
        }

    @property
    def name(self) -> str:
        return f"omni-ctc ({Path(self._run_ref).name})"

    @property
    def vocab(self) -> list[str]:
        return self._vocab

    @property
    def blank_index(self) -> int:
        return self._blank_index

    def _resolve_run_dir(self) -> Path:
        run_dir = Path(self._run_ref).expanduser().resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Omni run path does not exist: {run_dir}")
        return run_dir

    def load(self) -> None:
        if self._process is not None:
            return
        repo_root = Path(__file__).resolve().parents[4]
        project_root = repo_root / "projects" / "P003-compact-backbones"
        omni_root = (
            repo_root
            / "projects"
            / "P004-training-from-scratch"
            / "third_party"
            / "omnilingual-asr"
        )
        worker_script = project_root / "code" / "launch_omni_ctc_posterior_worker.py"
        resolved_uv = shutil.which("uv")
        if resolved_uv is None:
            raise RuntimeError("Could not resolve 'uv' on PATH.")
        cmd = [
            resolved_uv,
            "run",
            "--python",
            "3.12",
            "--with",
            "tbb>=2021.8",
            "--with-editable",
            str(omni_root),
            str(worker_script),
            "--run-dir",
            str(self._resolve_run_dir()),
            "--device",
            str(settings.torch_device),
        ]
        process = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=repo_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        ready_line = process.stdout.readline().strip()
        if not ready_line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            process.kill()
            raise RuntimeError(f"Omni worker failed to start.\n{stderr}")
        payload = json.loads(ready_line)
        if payload.get("status") != "ready":
            stderr = process.stderr.read() if process.stderr is not None else ""
            process.kill()
            raise RuntimeError(f"Omni worker startup failed: {payload}\n{stderr}")
        self._process = process
        logger.info(
            "Loaded %s: vocab=%d blank=%d device=%s",
            self.name,
            len(self._vocab),
            self._blank_index,
            payload.get("device"),
        )

    def unload(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(
                    json.dumps({"command": "shutdown", "request_id": "shutdown"})
                    + "\n"
                )
                process.stdin.flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Ignoring Omni worker shutdown write failure: %r", exc)
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            process.kill()
        self._process = None

    def _posterior_transport_dtype(self) -> np.dtype[np.float32] | np.dtype[np.float64]:
        dtype_name = settings.ctc_posterior_transport_dtype.lower()
        if dtype_name == "float32":
            return np.dtype(np.float32)
        if dtype_name == "float64":
            return np.dtype(np.float64)
        raise ValueError(
            "ctc_posterior_transport_dtype must be 'float32' or 'float64', "
            f"got {settings.ctc_posterior_transport_dtype!r}"
        )

    def get_posteriors_batch(
        self,
        audios: Sequence[np.ndarray],
        sample_rates: Sequence[int],
    ) -> list[np.ndarray]:
        if not audios:
            return []
        if len(set(sample_rates)) != 1:
            raise ValueError(
                "Batch requires a single sample rate, "
                f"got {sorted(set(sample_rates))}"
            )
        if sample_rates[0] != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"Omni backend expects {EXPECTED_SAMPLE_RATE}Hz audio, "
                f"got {sample_rates[0]}."
            )
        self.load()
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Omni worker not available.")

        request_id = uuid.uuid4().hex
        request_dir = self._request_root / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        wav_paths: list[str] = []
        for index, audio in enumerate(audios):
            path = request_dir / f"{index:04d}.wav"
            array = np.asarray(audio, dtype=np.float32)
            if array.ndim > 1:
                array = array.mean(axis=-1)
            sf.write(path, array, EXPECTED_SAMPLE_RATE)
            wav_paths.append(str(path))
        output_path = request_dir / "posteriors.npz"
        process.stdin.write(
            json.dumps(
                {
                    "command": "infer",
                    "request_id": request_id,
                    "wav_paths": wav_paths,
                    "output_path": str(output_path),
                }
            )
            + "\n"
        )
        process.stdin.flush()
        response_line = process.stdout.readline().strip()
        if not response_line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"Omni worker produced no response.\n{stderr}")
        response = json.loads(response_line)
        if response.get("status") != "ok":
            raise RuntimeError(f"Omni worker infer failed: {response}")
        data = np.load(output_path)
        transport_dtype = self._posterior_transport_dtype()
        return [
            np.asarray(data[f"p{index}"], dtype=transport_dtype)
            for index in range(len(audios))
        ]

    def get_posteriors(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.get_posteriors_batch([audio], [sample_rate])[0]

    def map_phone(self, arpabet_phone: str) -> list[int] | None:
        idx = self._phone_to_idx.get(arpabet_phone)
        return [idx] if idx is not None else None

"""Measure the real acoustic prefix length T' for our CTC (feeds gate 2).

T' = the CTC encoder's output frame count = the number of acoustic embeddings the NAR
editor would prepend (before any editor-side downsampling). We tap the production pipeline's
model-apply step (the same hook lm_decoding/run.py uses) to read `bl_out.seq_lens` per row,
paired with each clip's audio duration, and report the frame rate and the worst-case T'.

  uv run python experiments/nar/gate2b_audio_prefix_len.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET_ROOT = ROOT / "data/datasets/v3/version=0"
MODEL_CARD = "omni_ctc_300m_v2_tajik_v3_step_20000"
LANG = "tgk_Cyrl"
SR = 16000


def pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main() -> int:
    from fairseq2.nn import BatchLayout
    from omni_finetune_core.project import _load_test
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {MODEL_CARD}", flush=True)
    pipe = ASRInferencePipeline(MODEL_CARD, device=device, dtype=torch.float32)

    audio_all, _, corpora_all, _ = _load_test(DATASET_ROOT, LANG, 0, 40.0)
    print(f"loaded {len(audio_all)} rows total", flush=True)

    frames: list[int] = []  # captured in pipeline order (matches batch order)

    def apply_and_capture(batch):
        layout = BatchLayout(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )
        logits, bl_out = pipe.model(batch.source_seqs, layout)
        texts = []
        for j in range(logits.shape[0]):
            n = int(bl_out.seq_lens[j])
            frames.append(n)
            seq = logits[j, :n].argmax(dim=-1)
            mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
            mask[1:] = seq[1:] != seq[:-1]
            texts.append(pipe.token_decoder(seq[mask]))
        return texts

    pipe._apply_model_wav2vec2asr = apply_and_capture  # noqa: SLF001 — experiment tap

    splits = {"FLEURS test": "fleurs", "conversational held-out": "youtube-"}
    for title, prefix in splits.items():
        keep = [i for i, c in enumerate(corpora_all) if c.startswith(prefix)]
        audio = [audio_all[i] for i in keep]
        if not audio:
            print(f"\n## {title}: no rows — skip", flush=True)
            continue
        durs = np.array([len(a) / SR for a in audio])

        frames.clear()
        for start in range(0, len(audio), 100):
            pipe.transcribe(audio[start : start + 100], lang=[LANG] * len(audio[start:start+100]),
                            batch_size=8)
        T = np.array(frames)
        assert len(T) == len(audio), f"captured {len(T)} != {len(audio)} rows"

        fps = T / durs  # frames per second of audio
        print(f"\n## {title}  (n={len(audio)})", flush=True)
        print(f"  duration s   : p50 {pct(durs,50):.1f} | p95 {pct(durs,95):.1f} | "
              f"max {durs.max():.1f}", flush=True)
        print(f"  T' (frames)  : p50 {pct(T,50):.0f} | p95 {pct(T,95):.0f} | max {T.max()}",
              flush=True)
        print(f"  frame rate   : {np.median(fps):.1f} fps (≈ {1000/np.median(fps):.0f} ms/frame)",
              flush=True)
        print(f"  T' at max-dur 40s ≈ {40 * np.median(fps):.0f} frames", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

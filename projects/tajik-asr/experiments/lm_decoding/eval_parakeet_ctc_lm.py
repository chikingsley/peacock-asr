"""CTC-head WER for the fine-tuned 110M hybrid (TDT+CTC) parakeet: greedy vs CTC+KenLM.

Mirrors the omni lm_decoding methodology (run.py) but for the parakeet CTC head:
  - load the hybrid .nemo (fp32), switch to the CTC decoder
  - (a) greedy CTC WER via model.transcribe
  - (b) capture per-frame CTC log-probs, decode with pyctcdecode + a word KenLM
        (subword/BPE alphabet using sentencepiece '▁' word-boundary marker), small a/b sweep
  - WER via omni_curator.process.normalize(tgk_Cyrl) + omni_finetune_core.metrics.compute_measures

  uv run --project /home/simon/github/peacock-asr/projects/tajik-asr \
    python experiments/lm_decoding/eval_parakeet_ctc_lm.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEMO = ROOT / "data/parakeet/final/tajik-parakeet-tdt-110m_final.nemo"
LM_BIN = HERE / "lm4_parakeet.bin"
CORPUS = HERE / "corpus_parakeet_trainbig.txt"
LANG = "tgk_Cyrl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/parakeet/manifests/test_fleurs.jsonl"))
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--beam-width", type=int, default=64)
    args = ap.parse_args()

    from nemo.collections.asr.models import ASRModel
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures
    from pyctcdecode import build_ctcdecoder

    torch.set_num_threads(16)
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    if args.n:
        rows = rows[: args.n]
    paths = [r["audio_filepath"] for r in rows]
    refs = [r["text"] for r in rows]
    refs_norm = [normalize(r, LANG) for r in refs]
    print(f"rows: {len(rows)}  manifest: {Path(args.manifest).name}", flush=True)

    model = ASRModel.restore_from(str(NEMO), map_location=args.device)
    model = model.to(args.device).eval()
    model.change_decoding_strategy(decoder_type="ctc")  # use the CTC head
    print(f"loaded {NEMO.name}  ({type(model).__name__}, CTC head)", flush=True)

    # ---- (a) greedy CTC ----
    t0 = time.time()
    hyps = model.transcribe(paths, batch_size=args.bs)
    greedy_secs = time.time() - t0
    greedy = [h.text if hasattr(h, "text") else h for h in hyps]
    audio_secs = sum(float(r.get("duration", 0.0)) for r in rows)
    print(f"greedy transcribe: {greedy_secs:.1f}s ({audio_secs/greedy_secs:.0f}x RT)", flush=True)

    # ---- capture per-frame CTC log-probs for KenLM decoding ----
    # ctc decoding can return per-frame logprobs as hypothesis.alignments via the greedy decoder.
    cfg = model.cfg.decoding if hasattr(model.cfg, "decoding") else None
    from omegaconf import open_dict
    with open_dict(model.cfg.aux_ctc.decoding):
        model.cfg.aux_ctc.decoding.compute_timestamps = False
    model.change_decoding_strategy(decoder_type="ctc")
    hyps2 = model.transcribe(paths, batch_size=args.bs, return_hypotheses=True)
    captured: list[np.ndarray] = []
    for h in hyps2:
        lp = h.y_sequence if hasattr(h, "y_sequence") else None
        # NeMo CTC hypotheses carry per-frame logprobs in .alignments or .y_sequence depending on cfg;
        # safest: pull from h.alignments (T, V) when present.
        arr = None
        if getattr(h, "alignments", None) is not None and torch.is_tensor(h.alignments):
            arr = h.alignments
        elif lp is not None and torch.is_tensor(lp) and lp.dim() == 2:
            arr = lp
        if arr is None:
            raise RuntimeError("no per-frame logprobs on hypothesis; falling back needed")
        captured.append(arr.float().cpu().numpy())
    print(f"captured logprobs for {len(captured)} clips, shape0={captured[0].shape}", flush=True)

    # ---- build BPE alphabet for pyctcdecode ----
    sp = model.tokenizer.tokenizer  # underlying sentencepiece processor
    n_classes = captured[0].shape[-1]  # 1025 (1024 + blank)
    labels = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    # pyctcdecode subword: keep '▁' boundary marker; blank is the last index -> ""
    assert n_classes == len(labels) + 1, (n_classes, len(labels))
    labels = labels + [""]  # blank last

    unigrams = sorted({w for line in CORPUS.open() for w in line.split()})
    print(f"alphabet={len(labels)} (blank last)  unigrams={len(unigrams)}", flush=True)

    chunk_lps = [lp.astype(np.float32) for lp in captured]

    def score(name: str, out: list[str]) -> float:
        m = compute_measures(refs_norm, [normalize(h, LANG) for h in out])
        print(f"{name:<28} WER {m.wer:6.2f}%  CER {m.cer:6.2f}%", flush=True)
        return m.wer

    score("CTC greedy (NeMo)", greedy)

    grid = [(a, b) for a in (0.3, 0.5, 0.7) for b in (0.0, 1.0, 2.0)]
    best = None
    # Build ALL decoders first, THEN fork the pool so workers inherit the
    # populated pyctcdecode model_container (KenLM objects can't be pickled).
    decoders: list[tuple[str, object]] = [
        ("beam (no LM)", build_ctcdecoder(labels, kenlm_model_path=None, alpha=0.0, beta=0.0))
    ]
    for a, b in grid:
        decoders.append(
            (
                f"beam+KenLM a={a} b={b}",
                build_ctcdecoder(
                    labels, kenlm_model_path=str(LM_BIN), unigrams=unigrams, alpha=a, beta=b
                ),
            )
        )
    pool = mp.get_context("fork").Pool(8)
    try:
        for name, dec in decoders:
            out = dec.decode_batch(pool, chunk_lps, beam_width=args.beam_width)
            w = score(name, out)
            if name.startswith("beam+KenLM") and (best is None or w < best[0]):
                a, b = name.split("a=")[1].split(" b=")
                best = (w, float(a), float(b))
    finally:
        pool.terminate()

    print(f"\nBEST CTC+KenLM: WER {best[0]:.2f}%  (alpha={best[1]} beta={best[2]})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

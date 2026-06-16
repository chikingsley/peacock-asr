"""LM-fused beam decoding vs greedy CTC, same checkpoint (roadmap experiment #2).

The model is untouched: we wrap the pipeline's own model-apply step to capture per-frame
log-probabilities (preprocessing stays byte-identical to production greedy), then compare
three readouts on the same rows — greedy, plain beam search, and beam search fused with a
KenLM word 4-gram trained on our own training labels (corpus.txt -> lm4.bin via lmplz).
Reports WER/CER per readout plus decode-stage wall time, so the quality/speed trade is
measured rather than argued.

Rows stream through in chunks and log-probs are held in float16, so peak memory stays a few
GB regardless of row count (the 10,288-dim vocab makes whole-split float32 capture ~30 MB
per audio minute — the full v3 test split would need ~65 GB).

  uv run python experiments/lm_decoding/run.py --limit 50                        # smoke
  uv run python experiments/lm_decoding/run.py --only-corpus-prefix fleurs --sweep
  uv run python experiments/lm_decoding/run.py --only-corpus-prefix youtube-     # held-out
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LM_BIN = HERE / "lm4.bin"
TOKENIZER = ROOT / "src/tajik_asr/models/omniASR_tokenizer_written_v2.model"
DATASET_ROOT = ROOT / "data/datasets/v3/version=0"
MODEL_CARD = "omni_ctc_300m_v2_tajik_v3_step_20000"
LANG = "tgk_Cyrl"


def load_rows(limit: int, prefix: str | None) -> tuple[list, list[str]]:
    from omni_finetune_core.project import _load_test

    audio, refs, corpora, _ = _load_test(DATASET_ROOT, LANG, 0, 40.0)
    if prefix is not None:
        keep = [i for i, c in enumerate(corpora) if c.startswith(prefix)]
        audio = [audio[i] for i in keep]
        refs = [refs[i] for i in keep]
    if limit:
        audio, refs = audio[:limit], refs[:limit]
    return audio, refs


def build_labels(sp, captured: list[np.ndarray]) -> tuple[list[str], int]:
    """Alphabet from tokenizer pieces; CTC blank = the empirically dominant argmax index
    (blank rules silence). The tokenizer is character-level (10,284 single-char pieces,
    literal " " as the word boundary) — raw pieces ARE the alphabet; the multi-char specials
    are never emitted, so they get unused placeholder chars to keep the alphabet clean."""
    labels = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    counts = np.zeros(len(labels), dtype=np.int64)
    for lp in captured[:200]:
        idx, c = np.unique(lp.argmax(axis=-1), return_counts=True)
        counts[idx] += c
    blank = int(counts.argmax())
    print(f"blank index: {blank} (piece {labels[blank]!r})", flush=True)
    labels = [chr(0xE000 + i) if len(p) > 1 else p for i, p in enumerate(labels)]
    labels[blank] = ""
    return labels, blank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-corpus-prefix", default=None)
    ap.add_argument("--alpha", type=float, default=0.5, help="LM weight")
    ap.add_argument("--beta", type=float, default=1.5, help="word insertion bonus")
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-rows", type=int, default=100, help="rows per capture/decode chunk")
    ap.add_argument("--sweep", action="store_true", help="grid-search alpha/beta on these rows")
    args = ap.parse_args()

    import sentencepiece as spm
    from fairseq2.nn import BatchLayout
    from omni_curator.process import normalize
    from omni_finetune_core.metrics import compute_measures
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from pyctcdecode import build_ctcdecoder

    torch.set_num_threads(16)
    audio, refs = load_rows(args.limit, args.only_corpus_prefix)
    refs_norm = [normalize(r, LANG) for r in refs]
    print(f"rows: {len(audio)}", flush=True)

    pipe = ASRInferencePipeline(MODEL_CARD, device="cpu", dtype=torch.float32)

    # Capture log-probs by shadowing the pipeline's apply step with an identical body + tap.
    # float16: halves the buffer; the decoder gets float32 casts per chunk.
    captured: list[np.ndarray] = []

    def apply_and_capture(batch):  # noqa: ANN001, ANN202
        layout = BatchLayout(
            batch.source_seqs.shape,
            seq_lens=batch.source_seq_lens,
            device=batch.source_seqs.device,
        )
        logits, bl_out = pipe.model(batch.source_seqs, layout)
        texts = []
        for j in range(logits.shape[0]):
            n = bl_out.seq_lens[j]
            lp = torch.log_softmax(logits[j, :n].float(), dim=-1)
            captured.append(lp.to(torch.float16).cpu().numpy())
            seq = logits[j, :n].argmax(dim=-1)
            mask = torch.ones(seq.shape[0], dtype=torch.bool)
            mask[1:] = seq[1:] != seq[:-1]
            texts.append(pipe.token_decoder(seq[mask]))
        return texts

    pipe._apply_model_wav2vec2asr = apply_and_capture  # noqa: SLF001 — experiment tap

    grids: list[tuple[str, str | None, float, float]] = [
        ("beam (no LM)", None, 0.0, 0.0),
        (f"beam+LM a={args.alpha} b={args.beta}", str(LM_BIN), args.alpha, args.beta),
    ]
    if args.sweep:
        grids = [("beam (no LM)", None, 0.0, 0.0)] + [
            (f"beam+LM a={a} b={b}", str(LM_BIN), a, b)
            for a in (0.3, 0.5, 0.7, 1.0, 1.5)
            for b in (0.0, 1.5, 3.0)
        ]

    # Word vocabulary for the LM decoder (its binary KenLM can't self-report unigrams).
    unigrams = sorted({w for line in (HERE / "corpus.txt").open() for w in line.split()})
    print(f"unigrams: {len(unigrams)}", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    greedy_hyps: list[str] = []
    hyps: dict[str, list[str]] = {name: [] for name, *_ in grids}
    decode_secs: dict[str, float] = dict.fromkeys(hyps, 0.0)
    decoders = None
    pool = None
    total_frames = 0
    t_model = 0.0

    t_start = time.monotonic()
    try:
        for start in range(0, len(audio), args.chunk_rows):
            chunk = audio[start : start + args.chunk_rows]
            captured.clear()
            t0 = time.monotonic()
            greedy_hyps.extend(
                pipe.transcribe(chunk, lang=[LANG] * len(chunk), batch_size=args.batch_size)
            )
            t_model += time.monotonic() - t0
            total_frames += sum(lp.shape[0] for lp in captured)

            if decoders is None:
                labels, _ = build_labels(sp, captured)
                decoders = [
                    (
                        name,
                        build_ctcdecoder(
                            labels,
                            kenlm_model_path=kenlm_path,
                            unigrams=unigrams if kenlm_path else None,
                            alpha=alpha,
                            beta=beta,
                        ),
                    )
                    for name, kenlm_path, alpha, beta in grids
                ]
                # fork AFTER the decoders exist: workers inherit them (kenlm can't pickle)
                pool = mp.get_context("fork").Pool(8)

            chunk_lps = [lp.astype(np.float32) for lp in captured]
            for name, decoder in decoders:
                t0 = time.monotonic()
                hyps[name].extend(
                    decoder.decode_batch(pool, chunk_lps, beam_width=args.beam_width)
                )
                decode_secs[name] += time.monotonic() - t0
            done = min(start + args.chunk_rows, len(audio))
            rate = done / (time.monotonic() - t_start)
            eta = (len(audio) - done) / rate
            print(f"  chunk done: {done}/{len(audio)} rows "
                  f"({rate:.1f} rows/s, ETA {eta/60:.0f} min)", flush=True)
    finally:
        if pool is not None:
            pool.terminate()

    audio_secs = total_frames * 0.02  # ~20 ms per frame
    print(f"model forward+greedy: {t_model:.1f}s for ~{audio_secs/3600:.2f}h audio", flush=True)

    def score(name: str, out: list[str], seconds: float) -> None:
        m = compute_measures(refs_norm, [normalize(h, LANG) for h in out])
        print(f"{name:<22} WER {m.wer:6.2f}%  CER {m.cer:6.2f}%  "
              f"decode {seconds:6.1f}s ({audio_secs/max(seconds, 1e-9):7.0f}x realtime)",
              flush=True)

    score("greedy (production)", greedy_hyps, 0.0)
    for name, _ in decoders or []:
        score(name, hyps[name], decode_secs[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gate 1 for the NAR editor (NLE) feasibility check.

NLE interleaves ε around the N greedy-draft tokens -> 2N+1 output positions, and trains
with a CTC loss over those positions. A valid CTC alignment to the reference exists only if
the number of positions is at least the reference length: 2N+1 >= ref_len. Where
ref_len > 2N+1 the loss is unreachable / pathological for that example (the draft is so much
shorter than the truth that even filling every ε slot can't recover it).

This measures, per split, the share of rows with ref_len > 2N+1, plus the draft/ref length
distributions. Draft and reference are tokenized in the SAME omni vocab the editor would use
(shared tokenizer, size ~10288), normalized the same way training would normalize them.

  uv run python experiments/nar/gate1_draft_length.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOKENIZER = ROOT / "src/tajik_asr/models/omniASR_tokenizer_written_v2.model"
DATASET_ROOT = ROOT / "data/datasets/v3/version=0"
MODEL_CARD = "omni_ctc_300m_v2_tajik_v3_step_20000"
LANG = "tgk_Cyrl"


def pct(a: np.ndarray, p: float) -> float:
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main() -> int:
    import sentencepiece as spm
    from omni_curator.process import normalize
    from omni_finetune_core.project import _load_test
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  model: {MODEL_CARD}", flush=True)
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    pipe = ASRInferencePipeline(MODEL_CARD, device=device, dtype=torch.float32)

    audio_all, refs_all, corpora_all, _ = _load_test(DATASET_ROOT, LANG, 0, 40.0)
    print(f"loaded {len(audio_all)} rows total", flush=True)

    splits = {
        "FLEURS test": "fleurs",
        "conversational held-out": "youtube-",
    }

    for title, prefix in splits.items():
        keep = [i for i, c in enumerate(corpora_all) if c.startswith(prefix)]
        audio = [audio_all[i] for i in keep]
        refs = [refs_all[i] for i in keep]
        if not audio:
            print(f"\n## {title}: no rows match prefix {prefix!r} — skipping", flush=True)
            continue

        hyps: list[str] = []
        for start in range(0, len(audio), 100):
            chunk = audio[start : start + 100]
            hyps.extend(pipe.transcribe(chunk, lang=[LANG] * len(chunk), batch_size=8))
            print(f"  {title}: {min(start + 100, len(audio))}/{len(audio)}", flush=True)

        N = np.array([len(sp.encode(normalize(h, LANG))) for h in hyps])  # draft token len
        R = np.array([len(sp.encode(normalize(r, LANG))) for r in refs])  # ref token len
        slots = 2 * N + 1
        fail = slots < R  # CTC alignment impossible
        need_insert = R > N  # any net insertion needed at all

        print(f"\n## {title}  (n={len(audio)})", flush=True)
        print(f"  FAIL (ref_len > 2N+1): {int(fail.sum())} / {len(audio)} "
              f"= {100 * fail.mean():.2f}%", flush=True)
        print(f"  rows needing any insertion (ref_len > N): {int(need_insert.sum())} "
              f"= {100 * need_insert.mean():.1f}%", flush=True)
        print(f"  draft N tokens   : min {N.min()} | p50 {pct(N,50):.0f} | "
              f"p95 {pct(N,95):.0f} | max {N.max()}", flush=True)
        print(f"  ref   R tokens   : min {R.min()} | p50 {pct(R,50):.0f} | "
              f"p95 {pct(R,95):.0f} | max {R.max()}", flush=True)
        print(f"  slack (2N+1 - R) : min {int((slots-R).min())} | p1 {pct(slots-R,1):.0f} | "
              f"p50 {pct(slots-R,50):.0f}  (negative = FAIL)", flush=True)
        print(f"  R/(2N+1) ratio   : p50 {pct(R/slots,50):.2f} | p95 {pct(R/slots,95):.2f} | "
              f"max {pct(R/slots,100):.2f}", flush=True)
        print(f"  max seq len for editor (audio_prefix + 2N+1), R-side 2N+1 max = {slots.max()}",
              flush=True)
        if fail.any():
            idx = np.where(fail)[0][:5]
            print("  sample FAIL rows (draft || ref):", flush=True)
            for i in idx:
                print(f"    N={N[i]} R={R[i]}  {hyps[i][:60]!r} || {refs[i][:60]!r}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gate 4 feature extractor — multi-layer CTC encoder features for the richer projector.

Gate 3b's verdict was "conditioning, not recipe": a single Linear over the final CTC layer
can't drive corrections. IBM's NLE projector reads *several* encoder layers (their default:
4 layers) through a windowed Q-Former. So here we tap 4 intermediate layers of our 24-layer
CTC encoder (forward-hooks on `encoder.layers[...]`), and cache them stacked: [T', 4, 1024].
Draft + reference ids are unchanged from gate 3.

  uv run python experiments/nar/gate4_extract_features.py --n 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOKENIZER = ROOT.parents[1] / "base_models/omni/omniASR_tokenizer_written_v2.model"
DATASET_ROOT = ROOT / "data/datasets/v3/version=0"
MODEL_CARD = "omni_ctc_300m_v2_tajik_v3_step_20000"
LANG = "tgk_Cyrl"
OUT = HERE / "gate4_cache_fleurs.pt"
LAYER_INDICES = (5, 11, 17, 23)  # 0-indexed quarters of the 24-layer encoder (~IBM's 4 layers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--prefix", default="fleurs")
    args = ap.parse_args()

    import sentencepiece as spm
    from fairseq2.nn import BatchLayout
    from omni_curator.process import normalize
    from omni_finetune_core.project import _load_test
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    pipe = ASRInferencePipeline(MODEL_CARD, device=device, dtype=torch.float32)
    m = pipe.model
    n_enc = len(m.encoder.layers)
    print(f"device {device}  encoder layers {n_enc}  tapping {LAYER_INDICES}", flush=True)
    assert max(LAYER_INDICES) < n_enc, "layer index out of range"

    # forward-hooks capture each selected layer's output during the encoder forward
    cap: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_mod, _inp, out):
            cap[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for idx in LAYER_INDICES:
        m.encoder.layers[idx].register_forward_hook(make_hook(idx))

    audio_all, refs_all, corpora_all, _ = _load_test(DATASET_ROOT, LANG, 0, 40.0)
    keep = [i for i, c in enumerate(corpora_all) if c.startswith(args.prefix)][: args.n]
    print(f"using {len(keep)} rows (prefix {args.prefix!r})", flush=True)

    Hcap: dict[str, torch.Tensor] = {}

    def tap(batch):
        layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens,
                             device=batch.source_seqs.device)
        seqs, sl, _ = m.encoder_frontend.extract_features(batch.source_seqs, layout)
        seqs, _ = m.encoder_frontend.process_features(seqs, sl, None)
        cap.clear()
        h_final = m.encoder(seqs, sl)                     # triggers the layer hooks
        logits = m.final_proj(h_final)
        n = int(sl.seq_lens[0])
        # [T', num_layers, enc_dim]
        Hcap["H"] = torch.stack([cap[idx][0, :n] for idx in LAYER_INDICES], dim=1).half().cpu()
        seq = logits[0, :n].argmax(-1)
        mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
        mask[1:] = seq[1:] != seq[:-1]
        return [pipe.token_decoder(seq[mask])]

    pipe._apply_model_wav2vec2asr = tap  # noqa: SLF001

    rows = []
    for k in keep:
        Hcap.clear()
        draft_text = pipe.transcribe([audio_all[k]], lang=[LANG], batch_size=1)[0]
        rows.append({
            "H": Hcap["H"],                               # [T', num_layers, enc_dim]
            "draft_ids": torch.tensor(sp.encode(normalize(draft_text, LANG)), dtype=torch.long),
            "ref_ids": torch.tensor(sp.encode(normalize(refs_all[k], LANG)), dtype=torch.long),
            "draft_text": draft_text,
            "ref_text": refs_all[k],
        })
    torch.save({"layer_indices": list(LAYER_INDICES), "enc_dim": rows[0]["H"].shape[-1],
                "rows": rows}, OUT)
    Ts = [r["H"].shape[0] for r in rows]
    print(f"saved {len(rows)} rows -> {OUT.name}  | H shape {tuple(rows[0]['H'].shape)}  "
          f"T' min {min(Ts)} max {max(Ts)}", flush=True)
    print(f"  sample draft: {rows[0]['draft_text'][:70]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

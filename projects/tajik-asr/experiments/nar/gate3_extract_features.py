"""Extract frozen-CTC drafts + encoder features for the NAR learnability test.

Per row we cache: the CTC encoder hidden states H (acoustic features, [T', enc_dim]), the
greedy draft token ids, and the reference token ids — both tokenized with the shared omni
tokenizer (id space == the llama decoder's text_frontend). This is the precompute path the
build plan recommends (frozen CTC -> features identical every epoch).

  uv run python experiments/nar/gate3_extract_features.py --n 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOKENIZER = ROOT / "src/tajik_omnilingual_asr/models/omniASR_tokenizer_written_v2.model"
DATASET_ROOT = ROOT / "data/datasets/v3/version=0"
MODEL_CARD = "omni_ctc_300m_v2_tajik_v3_step_20000"
LANG = "tgk_Cyrl"
OUT = HERE / "gate3_cache_fleurs.pt"


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
    enc_dim = m.final_proj.weight.shape[1]
    print(f"device {device}  enc_dim {enc_dim}  vocab {sp.get_piece_size()}", flush=True)

    audio_all, refs_all, corpora_all, _ = _load_test(DATASET_ROOT, LANG, 0, 40.0)
    keep = [i for i, c in enumerate(corpora_all) if c.startswith(args.prefix)][: args.n]
    print(f"using {len(keep)} rows (prefix {args.prefix!r})", flush=True)

    # Tap the model-apply step so the pipeline does its own preprocessing (raw waveform ->
    # normalized features); we re-run the encoder there to grab hidden states H. Process one
    # row per transcribe() call so H aligns 1:1 with the reference (the pipeline may reorder
    # within a batch).
    cap: dict[str, torch.Tensor] = {}

    def tap(batch):
        layout = BatchLayout(batch.source_seqs.shape, seq_lens=batch.source_seq_lens,
                             device=batch.source_seqs.device)
        seqs, sl, _ = m.encoder_frontend.extract_features(batch.source_seqs, layout)
        seqs, _ = m.encoder_frontend.process_features(seqs, sl, None)
        H = m.encoder(seqs, sl)                           # [1, T', enc_dim]
        logits = m.final_proj(H)
        cap["H"] = H[0, : int(sl.seq_lens[0])].half().cpu()
        seq = logits[0, : int(sl.seq_lens[0])].argmax(-1)
        mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
        mask[1:] = seq[1:] != seq[:-1]
        return [pipe.token_decoder(seq[mask])]

    pipe._apply_model_wav2vec2asr = tap  # noqa: SLF001

    rows = []
    for k in keep:
        cap.clear()
        draft_text = pipe.transcribe([audio_all[k]], lang=[LANG], batch_size=1)[0]
        draft_ids = sp.encode(normalize(draft_text, LANG))
        ref_ids = sp.encode(normalize(refs_all[k], LANG))
        rows.append({
            "H": cap["H"],                                # [T', enc_dim]
            "draft_ids": torch.tensor(draft_ids, dtype=torch.long),
            "ref_ids": torch.tensor(ref_ids, dtype=torch.long),
            "draft_text": draft_text,
            "ref_text": refs_all[k],
        })
    torch.save({"enc_dim": enc_dim, "rows": rows}, OUT)
    Ts = [r["H"].shape[0] for r in rows]
    Ns = [len(r["draft_ids"]) for r in rows]
    Rs = [len(r["ref_ids"]) for r in rows]
    print(f"saved {len(rows)} rows -> {OUT.name}", flush=True)
    print(f"  T' (frames): min {min(Ts)} max {max(Ts)}", flush=True)
    print(f"  draft N    : min {min(Ns)} max {max(Ns)}", flush=True)
    print(f"  ref R      : min {min(Rs)} max {max(Rs)}", flush=True)
    print(f"  sample draft: {rows[0]['draft_text'][:70]!r}", flush=True)
    print(f"  sample ref  : {rows[0]['ref_text'][:70]!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

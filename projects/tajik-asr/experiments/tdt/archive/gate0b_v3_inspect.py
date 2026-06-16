"""Gate 0b — inspect parakeet-tdt-0.6b-v3 before any v3 training.

Answers the preflight questions: does v3 have an aux CTC head (→ is the CTC fallback inside
v3 or a separate model)? what are the TDT joint sizes (blank id, num_extra_outputs duration
bins) the restore code must respect? and does v3's 8192 unified tokenizer cover Tajik
Cyrillic or fall back to bytes/<unk>? Run in the farsi-asr NeMo env:

  uv run --project /home/simon/github/peacock-asr/projects/farsi-asr \
    python projects/tajik-asr/experiments/tdt/gate0b_v3_inspect.py
"""

from __future__ import annotations

# A few real Tajik (tgk_Cyrl) sentences to probe tokenizer coverage.
TAJIK = [
    "ин имконияти хуб барои дидани фаҷри қутбӣ аст зеро ки осмон мутаносиб аст",
    "салом чӣ хел шумо имрӯз кор мекунед",
    "номинатсияҳои беҳтарин филм коргардон ва наворбардорӣ ороиш",
]


def main() -> int:
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    print(f"\n=== v3 loaded: {type(model).__name__} ===", flush=True)

    # CTC head?
    has_ctc = hasattr(model, "ctc_decoder") and model.ctc_decoder is not None
    aux_ctc_cfg = "aux_ctc" in (model.cfg or {})
    print(f"aux CTC head present: {has_ctc}  (cfg.aux_ctc: {aux_ctc_cfg}) "
          f"-> CTC fallback is {'INSIDE v3' if has_ctc else 'a SEPARATE model'}", flush=True)

    # TDT joint sizes (restore code depends on these)
    j = model.joint
    print(f"joint.num_classes_with_blank = {j.num_classes_with_blank}", flush=True)
    print(f"joint.num_extra_outputs (TDT duration bins) = {j.num_extra_outputs}", flush=True)
    blank = j.num_classes_with_blank - 1 - j.num_extra_outputs
    print(f"=> real vocab tokens = {blank}, blank id = {blank}, "
          f"duration outputs occupy the top {j.num_extra_outputs}", flush=True)
    durs = getattr(getattr(model.cfg, "loss", {}), "get", lambda *_: None)("loss_name")
    print(f"loss/durations cfg: {model.cfg.get('model_defaults', {})}", flush=True)

    # Tokenizer coverage on Tajik
    tok = model.tokenizer
    vocab_size = getattr(tok, "vocab_size", None) or len(getattr(tok, "vocab", []))
    print(f"\ntokenizer: {type(tok).__name__}  vocab_size={vocab_size}", flush=True)
    unk = getattr(tok, "unk_id", None)
    for s in TAJIK:
        ids = tok.text_to_ids(s)
        pieces = tok.ids_to_tokens(ids) if hasattr(tok, "ids_to_tokens") else None
        n_unk = sum(1 for i in ids if unk is not None and i == unk)
        # byte-fallback pieces look like <0xNN>
        byte_pieces = sum(1 for p in (pieces or []) if isinstance(p, str) and p.startswith("<0x"))
        roundtrip = tok.ids_to_text(ids) if hasattr(tok, "ids_to_text") else "?"
        print(f"\n  text : {s[:50]}...", flush=True)
        print(f"  ids  : {len(ids)} tokens, {n_unk} <unk>, {byte_pieces} byte-fallback", flush=True)
        print(f"  rt   : {roundtrip[:50]}...  ({'OK' if roundtrip.strip()==s else 'LOSSY'})",
              flush=True)
        if pieces:
            print(f"  piece: {pieces[:16]}", flush=True)
    print("\n=== gate 0b done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

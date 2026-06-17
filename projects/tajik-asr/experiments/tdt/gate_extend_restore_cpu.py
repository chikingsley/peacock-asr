"""CPU dip-a-toe gates for the extend-restore recipe (safe while a GPU run is live).

Runs the four CPU gates from extend_restore_recipe.md on map_location=cpu:
  1. Tokenizer gate   — IDs 0..old identical; 12 singletons encode w/o <unk>; Tajik
                        train/dev round-trip; shared-Cyrillic sample identical old/new IDs.
  2. Shape gate       — after change_vocabulary: num_classes_with_blank==8198+K,
                        blank==8192+K, durations 8193+K..8197+K.
  3. Weight gate      — copied tensors torch.equal to base; new Tajik rows == post-
                        change_vocabulary random init (unchanged by restore).
  4. Logit gate       — fixed fp32 input: old-token + remapped blank/duration logits match
                        base v3 (max_abs_diff==0 direct rows, <=1e-6 full module path).

  uv run python projects/tajik-asr/experiments/tdt/gate_extend_restore_cpu.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel
from parakeet_finetune_core.extend_restore import (
    assert_shapes,
    expected_blank_and_durations,
    new_tajik_rows,
    restore_extended_decoder_joint,
    snapshot_decoder_joint,
    validate_restore,
)

HERE = Path(__file__).resolve().parent
PEACOCK = HERE.parents[3]
V3 = PEACOCK / "base_models/parakeet/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo"
TOK = PEACOCK / "projects/tajik-asr/data/parakeet/tokenizers/tok_extend_v3"
TRAIN = PEACOCK / "projects/tajik-asr/data/parakeet/manifests/train_big.jsonl"
DEV = PEACOCK / "projects/tajik-asr/data/parakeet/manifests/dev_big.jsonl"
OLD_VOCAB = 8192
SINGLETONS = list("ғқҳҷӯӣёҒҚҲҶӮӢЁ")
# a sentence using only shared Cyrillic (no Tajik-specific letters) -> identical old/new IDs
SHARED_CYRILLIC = "это простой текст на русском языке без таджикских букв"

_results: list[tuple[str, bool, str]] = []


def _record(name: str, detail: str, *, ok: bool) -> None:
    _results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _first_text(manifest: Path) -> str:
    with manifest.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                return json.loads(line)["text"]
    return ""


def tokenizer_gate(base_tok, ext_tok) -> None:
    # IDs 0..old identical
    ids_ok = all(
        base_tok.text_to_ids  # ensure attr exists
        and base_tok.ids_to_tokens([i])[0] == ext_tok.ids_to_tokens([i])[0]
        for i in range(OLD_VOCAB)
    )
    _record(
        "tokenizer:ids_0..8191_identical",
        "all 8192 pieces match" if ids_ok else "MISMATCH",
        ok=ids_ok,
    )

    unk = ext_tok.tokenizer.unk_id()
    no_unk = True
    for ch in SINGLETONS:
        ids = ext_tok.text_to_ids(ch)
        if any(i == unk for i in ids):
            no_unk = False
    _record(
        "tokenizer:marker_singletons_no_unk",
        f"all {len(SINGLETONS)} encode without <unk>",
        ok=no_unk,
    )

    tr_text = _first_text(TRAIN)
    dv_text = _first_text(DEV)
    tr_rt = ext_tok.ids_to_text(ext_tok.text_to_ids(tr_text))
    dv_rt = ext_tok.ids_to_text(ext_tok.text_to_ids(dv_text))
    # normalize whitespace for comparison
    tr_ok = tr_rt.split() == tr_text.split()
    dv_ok = dv_rt.split() == dv_text.split()
    _record(
        "tokenizer:tajik_train_roundtrip",
        f"train line rt={'OK' if tr_ok else 'LOSSY'}",
        ok=tr_ok,
    )
    _record(
        "tokenizer:tajik_dev_roundtrip",
        f"dev line rt={'OK' if dv_ok else 'LOSSY'}",
        ok=dv_ok,
    )

    base_ids = base_tok.text_to_ids(SHARED_CYRILLIC)
    ext_ids = ext_tok.text_to_ids(SHARED_CYRILLIC)
    shared_ok = base_ids == ext_ids
    _record(
        "tokenizer:shared_cyrillic_identical_ids",
        f"len={len(base_ids)} base==ext={shared_ok}",
        ok=shared_ok,
    )


def main() -> int:
    print("loading v3 (CPU)...", flush=True)
    base_model = ASRModel.restore_from(str(V3), map_location="cpu")
    base_model = base_model.cpu().eval()
    base_tok = base_model.tokenizer

    # build the model to be modified (separate instance so base stays untouched)
    model = ASRModel.restore_from(str(V3), map_location="cpu").cpu().eval()
    snapshot = snapshot_decoder_joint(model)

    print("change_vocabulary -> tok_extend_v3 (CPU)...", flush=True)
    model.change_vocabulary(new_tokenizer_dir=str(TOK), new_tokenizer_type="bpe")
    ext_tok = model.tokenizer

    # K from the resized joint
    extra = int(model.joint.num_extra_outputs)
    k = int(model.joint.num_classes_with_blank) - extra - 1 - OLD_VOCAB
    print(f"detected K={k}", flush=True)

    # capture random Tajik rows right after change_vocabulary (before restore)
    rand_rows = new_tajik_rows(model, old_vocab=OLD_VOCAB)

    # --- 1. tokenizer gate ---
    tokenizer_gate(base_tok, ext_tok)

    # --- 2. shape gate (post change_vocabulary, pre restore) ---
    try:
        assert_shapes(model, old_vocab=OLD_VOCAB, k=k, extra=extra)
        blank, durations = expected_blank_and_durations(OLD_VOCAB, k, extra)
        shape_ok = (
            int(model.joint.num_classes_with_blank) == OLD_VOCAB + k + 1 + extra
            and blank == OLD_VOCAB + k
            and tuple(durations) == tuple(range(OLD_VOCAB + k + 1, OLD_VOCAB + k + 1 + extra))
        )
        _record(
            "shape:num_classes/blank/durations",
            f"num_classes_with_blank={model.joint.num_classes_with_blank} blank={blank} "
            f"durations={list(durations)}",
            ok=shape_ok,
        )
    except AssertionError as exc:
        _record("shape:num_classes/blank/durations", str(exc), ok=False)

    # --- restore ---
    info = restore_extended_decoder_joint(model, snapshot, old_vocab=OLD_VOCAB)
    print(f"restore info: {info}", flush=True)

    # --- 3. weight gate ---
    new_blank = OLD_VOCAB + k
    block = 1 + extra
    dec_embed = dict(model.decoder.named_parameters())["prediction.embed.weight"]
    final_w = dict(model.joint.named_parameters())["joint_net.2.weight"]
    final_b = dict(model.joint.named_parameters())["joint_net.2.bias"]
    snap_embed = snapshot["decoder.prediction.embed.weight"]
    snap_w = snapshot["joint.joint_net.2.weight"]
    snap_b = snapshot["joint.joint_net.2.bias"]

    embed_tok_eq = torch.equal(dec_embed[:OLD_VOCAB].float().cpu(), snap_embed[:OLD_VOCAB])
    embed_blank_eq = torch.equal(dec_embed[new_blank].float().cpu(), snap_embed[OLD_VOCAB])
    w_tok_eq = torch.equal(final_w[:OLD_VOCAB].float().cpu(), snap_w[:OLD_VOCAB])
    w_block_eq = torch.equal(
        final_w[new_blank : new_blank + block].float().cpu(), snap_w[OLD_VOCAB : OLD_VOCAB + block]
    )
    b_block_eq = torch.equal(
        final_b[new_blank : new_blank + block].float().cpu(), snap_b[OLD_VOCAB : OLD_VOCAB + block]
    )
    copied_ok = embed_tok_eq and embed_blank_eq and w_tok_eq and w_block_eq and b_block_eq
    _record(
        "weight:copied_tensors_torch_equal",
        f"embed_tok={embed_tok_eq} embed_blank={embed_blank_eq} w_tok={w_tok_eq} "
        f"w_block={w_block_eq} b_block={b_block_eq}",
        ok=copied_ok,
    )

    # new Tajik rows unchanged from post-change_vocabulary random init
    cur_rows = new_tajik_rows(model, old_vocab=OLD_VOCAB)
    embed_rand_eq = torch.equal(cur_rows["embed_tajik"], rand_rows["embed_tajik"])
    joint_rand_eq = torch.equal(cur_rows["joint_tajik"], rand_rows["joint_tajik"])
    rand_ok = embed_rand_eq and joint_rand_eq
    _record(
        "weight:tajik_rows_unchanged_random",
        f"embed_rand_unchanged={embed_rand_eq} joint_rand_unchanged={joint_rand_eq}",
        ok=rand_ok,
    )

    # --- 4. logit-preservation gate ---
    gate = validate_restore(model, base_model, old_vocab=OLD_VOCAB)
    _record(
        "logit:direct_final_linear_rows",
        f"weight_max_abs_diff={gate['weight_max_abs_diff']} "
        f"bias_max_abs_diff={gate['bias_max_abs_diff']}",
        ok=gate["direct_pass"],
    )
    _record(
        "logit:full_module_path",
        f"path_max_abs_diff={gate['path_max_abs_diff']:.3e} (tol 1e-6)",
        ok=gate["path_pass"],
    )

    print("\n=== SUMMARY ===", flush=True)
    all_pass = all(ok for _, ok, _ in _results)
    for name, ok, _ in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    print(f"\nALL GATES: {'PASS' if all_pass else 'FAIL'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Neural N-best rescoring over persisted beam candidates (experiment A3).

Reads the top-N candidates a `farsi-omni-eval-lm --nbest` run persisted in the shared
benchmark store, scores every unique candidate with a frozen causal LM, and re-ranks with

    final = beam_score + alpha * neural_lm_logprob + beta * word_count

`--tune` grid-searches alpha/beta against the run's own references (dev discipline);
`--alpha/--beta` applies one frozen configuration. Neural scores are cached beside the
database so tuning never re-runs the LM.

Usage:
    uv run --no-sync python experiments/lm_decoding/rescore_neural.py \
        --run-id omni-ctc-300m-farsi-youtube_dev_conv-kenlm-a0.3-b0.0-beam64-nb16 --tune
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from farsi_asr import LANGUAGE, ROOT

DEFAULT_DB = ROOT / "data/benchmarks/results/persian-shortlist.sqlite3"
DEFAULT_MODEL = "HooshvareLab/gpt2-fa"
TUNE_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0)
TUNE_BETAS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--beam-score-field",
        choices=["lm", "acoustic"],
        default="lm",
        help="Which persisted beam score anchors the combination.",
    )
    return parser


def load_run(database: Path, run_id: str) -> tuple[dict[int, list], dict[int, str], float]:
    import sqlite3

    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT row_index, rank, hypothesis, acoustic_score, lm_score "
        "FROM nbest_candidates WHERE run_id = ? ORDER BY row_index, rank",
        (run_id,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"no n-best candidates for run {run_id!r}")
    candidates: dict[int, list] = defaultdict(list)
    for row_index, rank, text, acoustic, lm_score in rows:
        candidates[row_index].append((rank, text, acoustic, lm_score))
    refs = dict(
        db.execute(
            "SELECT row_index, reference FROM predictions WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    )
    audio_seconds = db.execute(
        "SELECT SUM(audio_seconds) FROM predictions WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    db.close()
    return candidates, refs, float(audio_seconds or 0.0)


def neural_scores(
    args: argparse.Namespace, texts: list[str], cache_path: Path
) -> tuple[dict[str, float], float]:
    cached: dict[str, float] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = sorted({text for text in texts if text not in cached and text.strip()})
    if not missing:
        return cached, 0.0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.to(args.device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(missing), args.batch_size):
            batch = missing[start : start + args.batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(args.device)
            logits = model(**enc).logits.float()
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].bool()
            token_scores = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for i, text in enumerate(batch):
                cached[text] = float(token_scores[i][mask[i]].sum())
            if (start // args.batch_size) % 50 == 0:
                print(f"  scored {start + len(batch)}/{len(missing)}", flush=True)
    score_secs = time.monotonic() - started
    cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return cached, score_secs


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    from omni_finetune_core.metrics import compute_measures

    kept = [(ref, hyp) for ref, hyp in pairs if ref.strip()]
    measures = compute_measures([r for r, _ in kept], [h for _, h in kept])
    return measures.wer


def rerank(
    candidates: dict[int, list],
    scores: dict[str, float],
    alpha: float,
    beta: float,
    field_index: int,
) -> dict[int, str]:
    chosen = {}
    for row_index, row in candidates.items():
        best_text, best_score = "", float("-inf")
        for _, text, acoustic, lm_score in row:
            beam_score = (acoustic, lm_score)[field_index]
            neural = scores.get(text, -1e9 if text.strip() else 0.0)
            final = beam_score + alpha * neural + beta * len(text.split())
            if final > best_score:
                best_score, best_text = final, text
        chosen[row_index] = best_text
    return chosen


def main(argv: list[str] | None = None) -> int:
    from omni_curator.process import normalize

    args = build_parser().parse_args(argv)
    if not args.tune and (args.alpha is None or args.beta is None):
        raise SystemExit("either --tune or both --alpha and --beta are required")

    candidates, refs, audio_seconds = load_run(args.database, args.run_id)
    texts = [text for row in candidates.values() for _, text, _, _ in row]
    print(f"{args.run_id}: {len(candidates)} rows, {len(set(texts))} unique candidates")

    cache_path = args.database.parent / f"neural_scores_{args.model.replace('/', '_')}.json"
    scores, score_secs = neural_scores(args, texts, cache_path)
    if score_secs:
        print(
            f"neural scoring: {score_secs:.1f}s for ~{audio_seconds / 3600:.2f}h audio "
            f"({audio_seconds / score_secs:.0f}xRT rescorer-only)",
            flush=True,
        )

    refs_norm = {i: normalize(ref, LANGUAGE) for i, ref in refs.items()}
    field_index = {"acoustic": 0, "lm": 1}[args.beam_score_field]

    baseline = {i: row[0][1] for i, row in candidates.items()}
    base_pairs = [(refs_norm[i], normalize(h, LANGUAGE)) for i, h in sorted(baseline.items())]
    base_wer = corpus_wer(base_pairs)
    print(f"baseline 1-best WER {base_wer:.2f}%")

    grid = (
        [(alpha, beta) for alpha in TUNE_ALPHAS for beta in TUNE_BETAS]
        if args.tune
        else [(args.alpha, args.beta)]
    )
    results = []
    for alpha, beta in grid:
        chosen = rerank(candidates, scores, alpha, beta, field_index)
        pairs = [(refs_norm[i], normalize(h, LANGUAGE)) for i, h in sorted(chosen.items())]
        wer = corpus_wer(pairs)
        results.append((wer, alpha, beta))
        print(f"alpha={alpha:<4} beta={beta:<5} WER {wer:6.2f}%", flush=True)

    best_wer, best_alpha, best_beta = min(results)
    print(
        f"\nbest: alpha={best_alpha} beta={best_beta} WER {best_wer:.2f}% "
        f"(baseline {base_wer:.2f}%, delta {base_wer - best_wer:+.2f})",
        flush=True,
    )
    print("RESCORE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

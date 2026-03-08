#!/usr/bin/env python3
"""Benchmark the exact weighted-k2 denominator path against the baseline loop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import k2  # noqa: E402

from p001_gop.gop import _ctc_forward_denom  # noqa: E402

from run_k2_denom_spike import (  # noqa: E402
    BLANK,
    build_unrolled_denom_graph_with_metadata,
    make_probs,
    make_seq,
)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_cases(
    utterances: int,
    vocab_size: int,
    frames: int,
    seq_len: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    cases: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    for utterance_idx in range(utterances):
        base_seed = seed + utterance_idx * 17
        params = make_probs(vocab_size, frames, base_seed)
        seq = make_seq(vocab_size, seq_len, base_seed + 1)
        for pos in range(seq_len):
            cases.append((params, seq, pos))
    return cases


def run_baseline(
    cases: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[float, list[float], list[float]]:
    start = time.perf_counter()
    denoms: list[float] = []
    occs: list[float] = []
    for params, seq, pos in cases:
        ll_denom, occ = _ctc_forward_denom(params, seq, pos=pos, blank=BLANK)
        denoms.append(ll_denom)
        occs.append(occ)
    elapsed = time.perf_counter() - start
    return elapsed, denoms, occs


def build_graph_batch(
    cases: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[
    float,
    list[tuple[str, dict[int, tuple[int, int, int]], int, int]],
]:
    start = time.perf_counter()
    graphs: list[tuple[str, dict[int, tuple[int, int, int]], int, int]] = []
    for params, seq, pos in cases:
        graph_text, metadata, _ = build_unrolled_denom_graph_with_metadata(
            params,
            seq,
            pos=pos,
            blank=BLANK,
        )
        graphs.append((graph_text, metadata, pos, graph_text.count("\n")))
    elapsed = time.perf_counter() - start
    return elapsed, graphs


def parse_graph_batch(
    graphs: list[tuple[str, dict[int, tuple[int, int, int]], int, int]],
    device: torch.device,
) -> tuple[float, k2.Fsa, list[dict[int, tuple[int, int, int]]], list[int], list[int]]:
    start = time.perf_counter()
    fsas = [k2.Fsa.from_str(graph_text, acceptor=True) for graph_text, _, _, _ in graphs]
    fsa_vec = k2.Fsa.from_fsas(fsas).to(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    metadata = [meta for _, meta, _, _ in graphs]
    positions = [pos for _, _, pos, _ in graphs]
    arc_counts = [arc_count for _, _, _, arc_count in graphs]
    return elapsed, fsa_vec, metadata, positions, arc_counts


def score_graph_batch(
    fsa_vec: k2.Fsa,
    device: torch.device,
) -> tuple[float, list[float]]:
    start = time.perf_counter()
    totals = fsa_vec.get_tot_scores(log_semiring=True, use_double_scores=True)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    denoms = (-totals).detach().cpu().tolist()
    return elapsed, [float(x) for x in denoms]


def occupancy_graph_batch(
    fsa_vec: k2.Fsa,
    metadata_batch: list[dict[int, tuple[int, int, int]]],
    positions: list[int],
    device: torch.device,
) -> tuple[float, list[float]]:
    start = time.perf_counter()
    forward_scores = fsa_vec.get_forward_scores(
        use_double_scores=True,
        log_semiring=True,
    )
    sync_if_needed(device)
    state_offsets = fsa_vec.arcs.row_splits(1).detach().cpu().tolist()

    occs: list[float] = []
    for case_idx, metadata in enumerate(metadata_batch):
        arb_state = 2 * positions[case_idx] + 1
        state_offset = state_offsets[case_idx]
        states_by_frame: dict[int, list[int]] = {}
        for state_id, (t, _, _) in metadata.items():
            global_state = state_offset + state_id
            states_by_frame.setdefault(t, []).append(global_state)

        occ = 0.0
        for states in states_by_frame.values():
            state_ix = torch.tensor(states, dtype=torch.long, device=device)
            log_z_t = torch.logsumexp(forward_scores[state_ix], dim=0).item()
            for global_state in states:
                _, logical_state, _ = metadata[global_state - state_offset]
                if logical_state == arb_state:
                    occ += float(torch.exp(forward_scores[global_state] - log_z_t).item())
        occs.append(occ)

    elapsed = time.perf_counter() - start
    return elapsed, occs


def summarize_diffs(reference: list[float], candidate: list[float]) -> dict[str, float]:
    diffs = [abs(a - b) for a, b in zip(reference, candidate, strict=True)]
    return {
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": sum(diffs) / len(diffs) if diffs else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-occupancy", action="store_true")
    args = parser.parse_args()

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    cases = make_cases(
        utterances=args.utterances,
        vocab_size=args.vocab_size,
        frames=args.frames,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    baseline_elapsed, baseline_denoms, baseline_occs = run_baseline(cases)
    build_elapsed, graphs = build_graph_batch(cases)
    parse_elapsed, fsa_vec, metadata_batch, positions, arc_counts = parse_graph_batch(
        graphs,
        device=device,
    )
    score_elapsed, k2_denoms = score_graph_batch(fsa_vec, device=device)

    if args.skip_occupancy:
        occ_elapsed = 0.0
        k2_occs: list[float] = []
        occ_summary = {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
    else:
        occ_elapsed, k2_occs = occupancy_graph_batch(
            fsa_vec,
            metadata_batch,
            positions,
            device=device,
        )
        occ_summary = summarize_diffs(baseline_occs, k2_occs)

    denom_summary = summarize_diffs(baseline_denoms, k2_denoms)

    result = {
        "device": str(device),
        "cases": len(cases),
        "utterances": args.utterances,
        "frames": args.frames,
        "vocab_size": args.vocab_size,
        "seq_len": args.seq_len,
        "avg_arcs_per_graph": sum(arc_counts) / len(arc_counts) if arc_counts else 0.0,
        "baseline_seconds": baseline_elapsed,
        "k2_build_seconds": build_elapsed,
        "k2_parse_seconds": parse_elapsed,
        "k2_score_seconds": score_elapsed,
        "k2_occupancy_seconds": occ_elapsed,
        "k2_total_seconds": build_elapsed + parse_elapsed + score_elapsed + occ_elapsed,
        "denom_parity": denom_summary,
        "occupancy_parity": occ_summary,
        "speedup_total_vs_baseline": (
            baseline_elapsed / (build_elapsed + parse_elapsed + score_elapsed + occ_elapsed)
            if (build_elapsed + parse_elapsed + score_elapsed + occ_elapsed) > 0
            else 0.0
        ),
        "speedup_score_only_vs_baseline": (
            baseline_elapsed / score_elapsed if score_elapsed > 0 else 0.0
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    denom_tol = 1e-5
    occ_tol = 1e-3
    if result["denom_parity"]["max_abs_diff"] > denom_tol:
        return 2
    if not args.skip_occupancy and result["occupancy_parity"]["max_abs_diff"] > occ_tol:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark the topology-only exact k2 denominator path."""

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

from benchmark_exact_k2_denom import (  # noqa: E402
    build_graph_batch,
    make_cases,
    occupancy_graph_batch,
    parse_graph_batch,
    run_baseline,
    score_graph_batch,
    summarize_diffs,
)
from run_k2_denom_spike import (  # noqa: E402
    BLANK,
    DenomTopology,
    build_unrolled_denom_topology,
)


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_topology_batch(
    cases: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[float, list[tuple[DenomTopology, int]]]:
    start = time.perf_counter()
    topologies: list[tuple[DenomTopology, int]] = []
    for params, seq, pos in cases:
        topology = build_unrolled_denom_topology(
            seq,
            pos=pos,
            frames=params.shape[1],
            vocab_size=params.shape[0],
            blank=BLANK,
        )
        topologies.append((topology, topology.graph_text.count("\n")))
    elapsed = time.perf_counter() - start
    return elapsed, topologies


def parse_topology_batch(
    topologies: list[tuple[DenomTopology, int]],
    device: torch.device,
) -> tuple[float, k2.Fsa, list[int], list[int]]:
    start = time.perf_counter()
    fsas = [topology.to_fsa() for topology, _ in topologies]
    fsa_vec = k2.Fsa.from_fsas(fsas).to(device)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    arc_counts = [arc_count for _, arc_count in topologies]
    return elapsed, fsa_vec, arc_counts


def intersect_topology_batch(
    fsa_vec: k2.Fsa,
    cases: list[tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
) -> tuple[float, k2.Fsa]:
    start = time.perf_counter()
    log_probs_batch = torch.stack(
        [
            torch.log(params.T.clamp(min=1e-30)).to(device=device, dtype=torch.float32)
            for params, _, _ in cases
        ],
        dim=0,
    )
    supervisions = torch.tensor(
        [[case_idx, 0, log_probs_batch.shape[1]] for case_idx in range(len(cases))],
        dtype=torch.int32,
    )
    dense = k2.DenseFsaVec(log_probs_batch, supervisions)
    lattice = k2.intersect_dense(
        fsa_vec,
        dense,
        output_beam=1_000_000.0,
        max_states=50_000_000,
        max_arcs=1_073_741_824,
        seqframe_idx_name="seqframe",
        frame_idx_name="frame",
    )
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return elapsed, lattice


def score_topology_batch(
    lattice: k2.Fsa,
    device: torch.device,
) -> tuple[float, list[float]]:
    start = time.perf_counter()
    totals = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    denoms = (-totals).detach().cpu().tolist()
    return elapsed, [float(x) for x in denoms]


def occupancy_topology_batch(
    lattice: k2.Fsa,
    topologies: list[tuple[DenomTopology, int]],
    device: torch.device,
) -> tuple[float, list[float]]:
    start = time.perf_counter()
    forward_scores = lattice.get_forward_scores(
        use_double_scores=True,
        log_semiring=True,
    )
    sync_if_needed(device)
    state_offsets = lattice.arcs.row_splits(1).detach().cpu().tolist()

    occs: list[float] = []
    for case_idx, (topology, _) in enumerate(topologies):
        occs.append(
            topology.occupancy_from_forward_scores(
                forward_scores,
                state_offset=state_offsets[case_idx],
            )
        )

    elapsed = time.perf_counter() - start
    return elapsed, occs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--with-occupancy", action="store_true")
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

    exact_build_elapsed, exact_graphs = build_graph_batch(cases)
    exact_parse_elapsed, exact_fsa_vec, exact_metadata, exact_positions, exact_arc_counts = (
        parse_graph_batch(exact_graphs, device=device)
    )
    exact_score_elapsed, exact_denoms = score_graph_batch(exact_fsa_vec, device=device)
    if not args.with_occupancy:
        exact_occ_elapsed = 0.0
        exact_occs: list[float] = []
    else:
        exact_occ_elapsed, exact_occs = occupancy_graph_batch(
            exact_fsa_vec,
            exact_metadata,
            exact_positions,
            device=device,
        )

    topology_build_elapsed, topologies = build_topology_batch(cases)
    topology_parse_elapsed, topology_fsa_vec, topology_arc_counts = parse_topology_batch(
        topologies,
        device=device,
    )
    topology_intersect_elapsed, topology_lattice = intersect_topology_batch(
        topology_fsa_vec,
        cases,
        device=device,
    )
    topology_score_elapsed, topology_denoms = score_topology_batch(
        topology_lattice,
        device=device,
    )
    if not args.with_occupancy:
        topology_occ_elapsed = 0.0
        topology_occs: list[float] = []
    else:
        topology_occ_elapsed, topology_occs = occupancy_topology_batch(
            topology_lattice,
            topologies,
            device=device,
        )

    exact_total = (
        exact_build_elapsed + exact_parse_elapsed + exact_score_elapsed + exact_occ_elapsed
    )
    topology_total = (
        topology_build_elapsed
        + topology_parse_elapsed
        + topology_intersect_elapsed
        + topology_score_elapsed
        + topology_occ_elapsed
    )

    result = {
        "device": str(device),
        "cases": len(cases),
        "utterances": args.utterances,
        "frames": args.frames,
        "vocab_size": args.vocab_size,
        "seq_len": args.seq_len,
        "baseline_seconds": baseline_elapsed,
        "exact_avg_arcs_per_graph": (
            sum(exact_arc_counts) / len(exact_arc_counts) if exact_arc_counts else 0.0
        ),
        "exact_build_seconds": exact_build_elapsed,
        "exact_parse_seconds": exact_parse_elapsed,
        "exact_score_seconds": exact_score_elapsed,
        "exact_occupancy_seconds": exact_occ_elapsed,
        "exact_total_seconds": exact_total,
        "exact_denom_parity": summarize_diffs(baseline_denoms, exact_denoms),
        "exact_occupancy_parity": (
            summarize_diffs(baseline_occs, exact_occs)
            if args.with_occupancy
            else {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
        ),
        "topology_avg_arcs_per_graph": (
            sum(topology_arc_counts) / len(topology_arc_counts) if topology_arc_counts else 0.0
        ),
        "topology_build_seconds": topology_build_elapsed,
        "topology_parse_seconds": topology_parse_elapsed,
        "topology_intersect_seconds": topology_intersect_elapsed,
        "topology_score_seconds": topology_score_elapsed,
        "topology_occupancy_seconds": topology_occ_elapsed,
        "topology_total_seconds": topology_total,
        "topology_denom_parity": summarize_diffs(baseline_denoms, topology_denoms),
        "topology_occupancy_parity": (
            summarize_diffs(baseline_occs, topology_occs)
            if args.with_occupancy
            else {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
        ),
        "topology_vs_exact_occupancy_parity": (
            summarize_diffs(exact_occs, topology_occs)
            if args.with_occupancy
            else {"max_abs_diff": 0.0, "mean_abs_diff": 0.0}
        ),
        "topology_speedup_total_vs_baseline": (
            baseline_elapsed / topology_total if topology_total > 0 else 0.0
        ),
        "topology_speedup_score_only_vs_baseline": (
            baseline_elapsed / topology_score_elapsed if topology_score_elapsed > 0 else 0.0
        ),
        "topology_speedup_total_vs_exact_total": (
            exact_total / topology_total if topology_total > 0 else 0.0
        ),
        "topology_speedup_score_only_vs_exact_score": (
            exact_score_elapsed / topology_score_elapsed
            if topology_score_elapsed > 0
            else 0.0
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    denom_tol = 1e-5
    occ_tol = 1e-3
    occ_match_tol = 1e-5
    if result["topology_denom_parity"]["max_abs_diff"] > denom_tol:
        return 2
    if (
        args.with_occupancy
        and result["topology_occupancy_parity"]["max_abs_diff"] > occ_tol
        and result["topology_vs_exact_occupancy_parity"]["max_abs_diff"] > occ_match_tol
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

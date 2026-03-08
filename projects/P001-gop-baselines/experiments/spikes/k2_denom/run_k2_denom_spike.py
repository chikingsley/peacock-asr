#!/usr/bin/env python3
"""Parity spike for replacing the scalar GOP denominator with k2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import k2  # noqa: E402

from p001_gop.gop import _ctc_forward, _ctc_forward_denom  # noqa: E402

BLANK = 0


@dataclass
class DenomTopology:
    graph_text: str
    state_metadata: dict[int, tuple[int, int, int]]
    final_state: int
    dest_frame: torch.Tensor
    dest_logical_state: torch.Tensor
    frame_state_ids: torch.Tensor
    frame_arb_mask: torch.Tensor

    def to_fsa(self, device: torch.device | None = None) -> k2.Fsa:
        fsa = k2.Fsa.from_str(self.graph_text, acceptor=True)
        fsa.dest_frame = self.dest_frame.clone()
        fsa.dest_logical_state = self.dest_logical_state.clone()
        fsa, arc_map = k2.arc_sort(fsa, ret_arc_map=True)
        fsa.dest_frame = fsa.dest_frame[arc_map]
        fsa.dest_logical_state = fsa.dest_logical_state[arc_map]
        if device is not None:
            fsa = fsa.to(device)
        return fsa

    def occupancy_from_forward_scores(
        self,
        forward_scores: torch.Tensor,
        state_offset: int = 0,
    ) -> float:
        state_ids = self.frame_state_ids.to(device=forward_scores.device, dtype=torch.long)
        valid_mask = state_ids >= 0
        global_state_ids = state_ids.masked_fill(~valid_mask, 0) + state_offset
        state_scores = forward_scores[global_state_ids]
        neg_inf = torch.tensor(
            float("-inf"),
            device=forward_scores.device,
            dtype=forward_scores.dtype,
        )
        state_scores = state_scores.masked_fill(~valid_mask, neg_inf)
        log_z = torch.logsumexp(state_scores, dim=1)
        probs = torch.exp(state_scores - log_z.unsqueeze(1))
        occ = probs.masked_select(self.frame_arb_mask.to(forward_scores.device) & valid_mask).sum()
        return float(occ.item())


def sort_arc_lines(arcs: list[str]) -> list[str]:
    def key(line: str) -> tuple[int, int, int]:
        parts = line.split()
        return int(parts[0]), int(parts[1]), int(parts[2])

    return sorted(arcs, key=key)


def add_arc(
    arcs: list[str],
    src: int,
    dst: int,
    score: float,
    label: int = 1,
) -> None:
    arcs.append(f"{src} {dst} {label} {score}")


def prune_and_serialize(arcs: list[str], final_state: int) -> str:
    parsed: list[tuple[int, int, int, float]] = []
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for line in arcs:
        src_s, dst_s, label_s, score_s = line.split()
        src = int(src_s)
        dst = int(dst_s)
        label = int(label_s)
        score = float(score_s)
        parsed.append((src, dst, label, score))
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    accessible = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in outgoing.get(node, []):
            if nxt not in accessible:
                accessible.add(nxt)
                stack.append(nxt)

    coaccessible = {final_state}
    stack = [final_state]
    while stack:
        node = stack.pop()
        for prev in incoming.get(node, []):
            if prev not in coaccessible:
                coaccessible.add(prev)
                stack.append(prev)

    keep = accessible & coaccessible
    ordered = sorted(keep - {0, final_state})
    remap = {0: 0}
    next_state = 1
    for old in ordered:
        remap[old] = next_state
        next_state += 1
    remap[final_state] = next_state

    kept_lines: list[str] = []
    for src, dst, label, score in parsed:
        if src in keep and dst in keep:
            kept_lines.append(f"{remap[src]} {remap[dst]} {label} {score}")
    return "\n".join([*sort_arc_lines(kept_lines), f"{remap[final_state]}"])


def prune_and_serialize_with_metadata(
    arcs: list[str],
    final_state: int,
    state_metadata: dict[int, tuple[int, int, int]],
) -> tuple[str, dict[int, tuple[int, int, int]], int]:
    parsed: list[tuple[int, int, int, float]] = []
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for line in arcs:
        src_s, dst_s, label_s, score_s = line.split()
        src = int(src_s)
        dst = int(dst_s)
        label = int(label_s)
        score = float(score_s)
        parsed.append((src, dst, label, score))
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    accessible = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in outgoing.get(node, []):
            if nxt not in accessible:
                accessible.add(nxt)
                stack.append(nxt)

    coaccessible = {final_state}
    stack = [final_state]
    while stack:
        node = stack.pop()
        for prev in incoming.get(node, []):
            if prev not in coaccessible:
                coaccessible.add(prev)
                stack.append(prev)

    keep = accessible & coaccessible
    ordered = sorted(keep - {0, final_state})
    remap = {0: 0}
    next_state = 1
    for old in ordered:
        remap[old] = next_state
        next_state += 1
    remap[final_state] = next_state

    kept_lines: list[str] = []
    kept_metadata: dict[int, tuple[int, int, int]] = {}
    for src, dst, label, score in parsed:
        if src in keep and dst in keep:
            kept_lines.append(f"{remap[src]} {remap[dst]} {label} {score}")
    for state_id, metadata in state_metadata.items():
        if state_id in keep:
            kept_metadata[remap[state_id]] = metadata
    return (
        "\n".join([*sort_arc_lines(kept_lines), f"{remap[final_state]}"]),
        kept_metadata,
        remap[final_state],
    )


def prune_topology_records(
    arcs: list[tuple[int, int, int]],
    final_state: int,
    state_metadata: dict[int, tuple[int, int, int]],
    final_frame: int,
) -> DenomTopology:
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for src, dst, _ in arcs:
        outgoing.setdefault(src, []).append(dst)
        incoming.setdefault(dst, []).append(src)

    accessible = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in outgoing.get(node, []):
            if nxt not in accessible:
                accessible.add(nxt)
                stack.append(nxt)

    coaccessible = {final_state}
    stack = [final_state]
    while stack:
        node = stack.pop()
        for prev in incoming.get(node, []):
            if prev not in coaccessible:
                coaccessible.add(prev)
                stack.append(prev)

    keep = accessible & coaccessible
    ordered = sorted(keep - {0, final_state})
    remap = {0: 0}
    next_state = 1
    for old in ordered:
        remap[old] = next_state
        next_state += 1
    remap[final_state] = next_state

    kept_lines: list[str] = []
    kept_state_metadata: dict[int, tuple[int, int, int]] = {}
    dest_frames: list[int] = []
    dest_logical_states: list[int] = []
    for src, dst, label in sorted(arcs, key=lambda item: (item[0], item[2], item[1])):
        if src not in keep or dst not in keep:
            continue
        kept_lines.append(f"{remap[src]} {remap[dst]} {label} 0.0")
        if dst == final_state:
            dest_frames.append(final_frame)
            dest_logical_states.append(-1)
        else:
            frame_idx, logical_state, _ = state_metadata[dst]
            dest_frames.append(frame_idx)
            dest_logical_states.append(logical_state)
    for state_id, metadata in state_metadata.items():
        if state_id in keep:
            kept_state_metadata[remap[state_id]] = metadata
    return DenomTopology(
        graph_text="\n".join([*kept_lines, f"{remap[final_state]}"]),
        state_metadata=kept_state_metadata,
        final_state=remap[final_state],
        dest_frame=torch.tensor(dest_frames, dtype=torch.int32),
        dest_logical_state=torch.tensor(dest_logical_states, dtype=torch.int32),
        frame_state_ids=torch.empty((0, 0), dtype=torch.int32),
        frame_arb_mask=torch.empty((0, 0), dtype=torch.bool),
    )


def build_occupancy_layout(
    state_metadata: dict[int, tuple[int, int, int]],
    pos: int,
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states_by_frame: dict[int, list[int]] = defaultdict(list)
    for state_id, (frame_idx, _, _) in state_metadata.items():
        states_by_frame[frame_idx].append(state_id)

    max_states = max((len(states) for states in states_by_frame.values()), default=0)
    frame_state_ids = torch.full((frames, max_states), -1, dtype=torch.int32)
    frame_arb_mask = torch.zeros((frames, max_states), dtype=torch.bool)
    arb_state = 2 * pos + 1
    for frame_idx in range(frames):
        states = sorted(states_by_frame.get(frame_idx, []))
        for col_idx, state_id in enumerate(states):
            frame_state_ids[frame_idx, col_idx] = state_id
            _, logical_state, _ = state_metadata[state_id]
            frame_arb_mask[frame_idx, col_idx] = logical_state == arb_state
    return frame_state_ids, frame_arb_mask


def make_probs(vocab_size: int, frames: int, seed: int) -> torch.Tensor:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    logits = torch.randn(vocab_size, frames, generator=rng, dtype=torch.float64)
    return torch.softmax(logits, dim=0)


def make_seq(vocab_size: int, seq_len: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    seq = rng.integers(1, vocab_size, size=seq_len, endpoint=False)
    return torch.tensor(seq, dtype=torch.long)


def build_denom_graph(seq: list[int], pos: int, vocab_size: int, blank: int = BLANK) -> str:
    """Build an explicit acceptor matching _ctc_forward_denom() semantics.

    The arbitrary position is expanded into one state per allowed phone. This is
    the graph equivalent of the [L, T, V] alpha tensor in the current baseline.
    """

    next_token = seq[pos + 1] if pos < len(seq) - 1 else None
    allowed = [tok for tok in range(vocab_size) if tok != blank and tok != next_token]
    state_id = 0
    blank_states: dict[int, int] = {}
    token_states: dict[int, int] = {}
    arb_states: dict[int, int] = {}

    for j in range(len(seq) + 1):
        blank_states[j] = state_id
        state_id += 1
        if j < len(seq):
            if j == pos:
                for tok in allowed:
                    arb_states[tok] = state_id
                    state_id += 1
            else:
                token_states[j] = state_id
                state_id += 1

    final_state = state_id
    arcs: list[str] = []

    # Blank states: self loop on blank, predecessor from prior token/arbitrary.
    for j, b_state in blank_states.items():
        arcs.append(f"{b_state} {b_state} {blank} 0.0")
        if j == 0:
            continue
        prev_idx = j - 1
        if prev_idx == pos:
            for arb_state in arb_states.values():
                arcs.append(f"{arb_state} {b_state} {blank} 0.0")
        else:
            arcs.append(f"{token_states[prev_idx]} {b_state} {blank} 0.0")

    # Standard token states, with the special "after arbitrary" case encoded explicitly.
    for j, t_state in token_states.items():
        token = seq[j]
        arcs.append(f"{t_state} {t_state} {token} 0.0")
        arcs.append(f"{blank_states[j]} {t_state} {token} 0.0")

        if j == 0:
            continue

        if j == pos + 1:
            arcs.append(f"{blank_states[pos]} {t_state} {token} 0.0")
            if pos > 0 and seq[pos - 1] != token:
                arcs.append(f"{token_states[pos - 1]} {t_state} {token} 0.0")
            for arb_token, arb_state in arb_states.items():
                if arb_token != token:
                    arcs.append(f"{arb_state} {t_state} {token} 0.0")
            continue

        prev_idx = j - 1
        if prev_idx != pos and seq[prev_idx] != token:
            arcs.append(f"{token_states[prev_idx]} {t_state} {token} 0.0")

    # Arbitrary substates.
    for token, arb_state in arb_states.items():
        arcs.append(f"{blank_states[pos]} {arb_state} {token} 0.0")
        if pos > 0 and seq[pos - 1] != token:
            arcs.append(f"{token_states[pos - 1]} {arb_state} {token} 0.0")
        for prev_token, prev_state in arb_states.items():
            if prev_token != token:
                arcs.append(f"{prev_state} {arb_state} {token} 0.0")

    # Final arcs: standard CTC finalization from the last blank and last token family.
    arcs.append(f"{blank_states[len(seq)]} {final_state} -1 0.0")
    if pos == len(seq) - 1:
        for arb_state in arb_states.values():
            arcs.append(f"{arb_state} {final_state} -1 0.0")
    else:
        arcs.append(f"{token_states[len(seq) - 1]} {final_state} -1 0.0")

    return prune_and_serialize(arcs, final_state)


def fsa_nll(log_probs: torch.Tensor, fsa: k2.Fsa, device: torch.device) -> float:
    fsa = fsa.to(device)
    dense = k2.DenseFsaVec(
        log_probs.unsqueeze(0),
        torch.tensor([[0, 0, log_probs.shape[0]]], dtype=torch.int32),
    )
    lattice = k2.intersect_dense(
        fsa,
        dense,
        output_beam=1_000_000.0,
        max_states=50_000_000,
        max_arcs=1_073_741_824,
    )
    total = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
    return -float(total[0].item())


def weighted_fsa_nll(graph_text: str) -> float:
    fsa = k2.Fsa.from_str(graph_text, acceptor=True)
    fsa_vec = k2.Fsa.from_fsas([fsa])
    total = fsa_vec.get_tot_scores(log_semiring=True, use_double_scores=True)
    return -float(total[0].item())


def build_unrolled_ctc_graph(params: torch.Tensor, seq: torch.Tensor, blank: int = BLANK) -> str:
    frames = params.shape[1]
    seq_len = int(seq.shape[0])
    big_l = 2 * seq_len + 1

    state_ids: dict[tuple[int, int], int] = {}
    next_id = 1
    arcs: list[str] = []

    def get_state(t: int, s: int) -> int:
        nonlocal next_id
        key = (t, s)
        if key not in state_ids:
            state_ids[key] = next_id
            next_id += 1
        return state_ids[key]

    def maybe_add_curr(curr_s: int, src_states: list[int], weight: float) -> None:
        valid_src: list[int] = []
        for prev_s in src_states:
            if (t - 1, prev_s) in state_ids:
                valid_src.append(prev_s)
        if not valid_src:
            return
        curr = get_state(t, curr_s)
        for prev_s in valid_src:
            add_arc(arcs, get_state(t - 1, prev_s), curr, weight)

    # Initial frame.
    add_arc(arcs, 0, get_state(0, 0), math.log(float(params[blank, 0].item())))
    add_arc(arcs, 0, get_state(0, 1), math.log(float(params[int(seq[0].item()), 0].item())))

    for t in range(1, frames):
        start = max(0, big_l - 1 - 2 * (frames - t))
        for s in range(start, big_l):
            li = (s - 1) // 2
            if s % 2 == 0:
                weight = math.log(float(params[blank, t].item()))
                src = [s]
                if s != 0:
                    src.append(s - 1)
                maybe_add_curr(s, src, weight)
            else:
                token = int(seq[li].item())
                weight = math.log(float(params[token, t].item()))
                src = [s, s - 1]
                if s != 1 and int(seq[li].item()) != int(seq[li - 1].item()):
                    src.append(s - 2)
                maybe_add_curr(s, src, weight)

    final_state = next_id
    for s in range(big_l):
        key = (frames - 1, s)
        if key in state_ids:
            add_arc(arcs, state_ids[key], final_state, 0.0, label=-1)
    return prune_and_serialize(arcs, final_state)


def build_unrolled_denom_graph(
    params: torch.Tensor,
    seq: torch.Tensor,
    pos: int,
    blank: int = BLANK,
) -> str:
    graph_text, _, _ = build_unrolled_denom_graph_with_metadata(
        params,
        seq,
        pos,
        blank=blank,
    )
    return graph_text


def build_unrolled_denom_graph_with_metadata(
    params: torch.Tensor,
    seq: torch.Tensor,
    pos: int,
    blank: int = BLANK,
) -> tuple[str, dict[int, tuple[int, int, int]], int]:
    frames = params.shape[1]
    seq_len = int(seq.shape[0])
    big_l = 2 * seq_len + 1
    big_p = params.shape[0]
    nli = int(seq[pos + 1].item()) if pos < seq_len - 1 else None

    state_ids: dict[tuple[int, int, int], int] = {}
    state_metadata: dict[int, tuple[int, int, int]] = {}
    next_id = 1
    arcs: list[str] = []

    def get_state(t: int, s: int, p: int = 0) -> int:
        nonlocal next_id
        key = (t, s, p)
        if key not in state_ids:
            state_ids[key] = next_id
            state_metadata[next_id] = key
            next_id += 1
        return state_ids[key]

    def allowed_arb_tokens(exclude: set[int]) -> list[int]:
        out: list[int] = []
        for token in range(big_p):
            if token not in exclude:
                out.append(token)
        return out

    def maybe_add_curr(curr_s: int, curr_p: int, preds: list[tuple[int, int]], weight: float) -> None:
        valid_preds: list[tuple[int, int]] = []
        for prev_s, prev_p in preds:
            if (t - 1, prev_s, prev_p) in state_ids:
                valid_preds.append((prev_s, prev_p))
        if not valid_preds:
            return
        curr = get_state(t, curr_s, curr_p)
        for prev_s, prev_p in valid_preds:
            add_arc(arcs, get_state(t - 1, prev_s, prev_p), curr, weight)

    # Initial frame.
    add_arc(arcs, 0, get_state(0, 0, 0), math.log(float(params[blank, 0].item())))
    if pos == 0:
        for token in allowed_arb_tokens({blank} | ({nli} if nli is not None else set())):
            add_arc(arcs, 0, get_state(0, 1, token), math.log(float(params[token, 0].item())))
        if seq_len > 1:
            token = int(seq[1].item())
            add_arc(arcs, 0, get_state(0, 3, 0), math.log(float(params[token, 0].item())))
    else:
        add_arc(
            arcs,
            0,
            get_state(0, 1, 0),
            math.log(float(params[int(seq[0].item()), 0].item())),
        )

    for t in range(1, frames):
        lo = big_l - 1 - 2 * (frames - t)
        if (lo - 1) / 2 == pos:
            lo -= 2
        start = max(0, lo)

        for s in range(start, big_l):
            li = (s - 1) // 2

            if s % 2 == 0:
                weight = math.log(float(params[blank, t].item()))
                preds: list[tuple[int, int]] = [(s, 0)]
                arb_idx = s - 1
                if (arb_idx - 1) / 2 == pos:
                    zero = {blank}
                    if nli is not None:
                        zero.add(nli)
                    for token in allowed_arb_tokens(zero):
                        preds.append((arb_idx, token))
                elif s != 0:
                    preds.append((s - 1, 0))
                maybe_add_curr(s, 0, preds, weight)
                continue

            if pos != li and pos != li - 1:
                token_id = int(seq[li].item())
                weight = math.log(float(params[token_id, t].item()))
                preds = [(s, 0), (s - 1, 0)]
                if s != 1 and int(seq[li].item()) != int(seq[li - 1].item()):
                    preds.append((s - 2, 0))
                maybe_add_curr(s, 0, preds, weight)
                continue

            if pos == li - 1:
                token_id = int(seq[li].item())
                weight = math.log(float(params[token_id, t].item()))
                preds = [(s, 0), (s - 1, 0)]

                zero = {blank, token_id}
                if nli is not None:
                    zero.add(nli)
                for token in allowed_arb_tokens(zero):
                    preds.append((s - 2, token))

                preds.append((s - 3, 0))
                if li - 2 >= 0 and int(seq[li - 2].item()) != token_id:
                    preds.append((s - 4, 0))
                maybe_add_curr(s, 0, preds, weight)
                continue

            # Arbitrary state.
            base_exclude = {blank}
            if nli is not None:
                base_exclude.add(nli)
            for token in allowed_arb_tokens(base_exclude):
                weight = math.log(float(params[token, t].item()))
                preds: list[tuple[int, int]] = [(s, token)]
                preds.append((s - 1, 0))
                if s != 1 and token != int(seq[li - 1].item()):
                    preds.append((s - 2, 0))
                maybe_add_curr(s, token, preds, weight)

    final_state = next_id
    for key, node_id in state_ids.items():
        if key[0] == frames - 1:
            add_arc(arcs, node_id, final_state, 0.0, label=-1)
    return prune_and_serialize_with_metadata(arcs, final_state, state_metadata)


def build_unrolled_denom_topology(
    seq: torch.Tensor,
    pos: int,
    frames: int,
    vocab_size: int,
    blank: int = BLANK,
) -> DenomTopology:
    seq_len = int(seq.shape[0])
    big_l = 2 * seq_len + 1
    nli = int(seq[pos + 1].item()) if pos < seq_len - 1 else None

    state_ids: dict[tuple[int, int, int], int] = {}
    state_metadata: dict[int, tuple[int, int, int]] = {}
    next_id = 1
    arcs: list[tuple[int, int, int]] = []

    def get_state(t: int, s: int, p: int = 0) -> int:
        nonlocal next_id
        key = (t, s, p)
        if key not in state_ids:
            state_ids[key] = next_id
            state_metadata[next_id] = key
            next_id += 1
        return state_ids[key]

    def allowed_arb_tokens(exclude: set[int]) -> list[int]:
        out: list[int] = []
        for token in range(vocab_size):
            if token not in exclude:
                out.append(token)
        return out

    def maybe_add_curr(curr_s: int, curr_p: int, preds: list[tuple[int, int]], label: int) -> None:
        valid_preds: list[tuple[int, int]] = []
        for prev_s, prev_p in preds:
            if (t - 1, prev_s, prev_p) in state_ids:
                valid_preds.append((prev_s, prev_p))
        if not valid_preds:
            return
        curr = get_state(t, curr_s, curr_p)
        for prev_s, prev_p in valid_preds:
            arcs.append((get_state(t - 1, prev_s, prev_p), curr, label))

    arcs.append((0, get_state(0, 0, 0), blank))
    if pos == 0:
        for token in allowed_arb_tokens({blank} | ({nli} if nli is not None else set())):
            arcs.append((0, get_state(0, 1, token), token))
        if seq_len > 1:
            token = int(seq[1].item())
            arcs.append((0, get_state(0, 3, 0), token))
    else:
        arcs.append((0, get_state(0, 1, 0), int(seq[0].item())))

    for t in range(1, frames):
        lo = big_l - 1 - 2 * (frames - t)
        if (lo - 1) / 2 == pos:
            lo -= 2
        start = max(0, lo)

        for s in range(start, big_l):
            li = (s - 1) // 2

            if s % 2 == 0:
                preds: list[tuple[int, int]] = [(s, 0)]
                arb_idx = s - 1
                if (arb_idx - 1) / 2 == pos:
                    zero = {blank}
                    if nli is not None:
                        zero.add(nli)
                    for token in allowed_arb_tokens(zero):
                        preds.append((arb_idx, token))
                elif s != 0:
                    preds.append((s - 1, 0))
                maybe_add_curr(s, 0, preds, blank)
                continue

            if pos != li and pos != li - 1:
                token_id = int(seq[li].item())
                preds = [(s, 0), (s - 1, 0)]
                if s != 1 and int(seq[li].item()) != int(seq[li - 1].item()):
                    preds.append((s - 2, 0))
                maybe_add_curr(s, 0, preds, token_id)
                continue

            if pos == li - 1:
                token_id = int(seq[li].item())
                preds = [(s, 0), (s - 1, 0)]

                zero = {blank, token_id}
                if nli is not None:
                    zero.add(nli)
                for token in allowed_arb_tokens(zero):
                    preds.append((s - 2, token))

                preds.append((s - 3, 0))
                if li - 2 >= 0 and int(seq[li - 2].item()) != token_id:
                    preds.append((s - 4, 0))
                maybe_add_curr(s, 0, preds, token_id)
                continue

            base_exclude = {blank}
            if nli is not None:
                base_exclude.add(nli)
            for token in allowed_arb_tokens(base_exclude):
                preds: list[tuple[int, int]] = [(s, token), (s - 1, 0)]
                if s != 1 and token != int(seq[li - 1].item()):
                    preds.append((s - 2, 0))
                maybe_add_curr(s, token, preds, token)

    final_state = next_id
    for key, node_id in state_ids.items():
        if key[0] == frames - 1:
            arcs.append((node_id, final_state, -1))
    topology = prune_topology_records(
        arcs,
        final_state,
        state_metadata,
        final_frame=frames,
    )
    frame_state_ids, frame_arb_mask = build_occupancy_layout(
        topology.state_metadata,
        pos=pos,
        frames=frames,
    )
    topology.frame_state_ids = frame_state_ids
    topology.frame_arb_mask = frame_arb_mask
    return topology


def intersect_unrolled_denom_topology(
    topology: DenomTopology,
    log_probs: torch.Tensor,
    device: torch.device,
) -> k2.Fsa:
    fsa = topology.to_fsa(device)
    dense = k2.DenseFsaVec(
        log_probs.unsqueeze(0),
        torch.tensor([[0, 0, log_probs.shape[0]]], dtype=torch.int32),
    )
    return k2.intersect_dense(
        k2.Fsa.from_fsas([fsa]),
        dense,
        output_beam=1_000_000.0,
        max_states=50_000_000,
        max_arcs=1_073_741_824,
        seqframe_idx_name="seqframe",
        frame_idx_name="frame",
    )


def topology_denom_nll(lattice: k2.Fsa) -> float:
    total = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
    return -float(total[0].item())


def topology_denom_occupancy(lattice: k2.Fsa, pos: int) -> float:
    del pos
    raise NotImplementedError("Use DenomTopology.occupancy_from_forward_scores().")


def unrolled_denom_occupancy_k2(
    graph_text: str,
    state_metadata: dict[int, tuple[int, int, int]],
    pos: int,
    device: torch.device,
) -> float:
    fsa = k2.Fsa.from_str(graph_text, acceptor=True).to(device)
    fsa_vec = k2.Fsa.from_fsas([fsa])
    forward_scores = fsa_vec.get_forward_scores(
        use_double_scores=True,
        log_semiring=True,
    )

    states_by_frame: dict[int, list[int]] = defaultdict(list)
    for state_id, (t, _, _) in state_metadata.items():
        states_by_frame[t].append(state_id)

    occ = 0.0
    arb_state = 2 * pos + 1
    for t, states in states_by_frame.items():
        state_ix = torch.tensor(states, dtype=torch.long, device=forward_scores.device)
        log_z_t = torch.logsumexp(forward_scores[state_ix], dim=0).item()
        for state_id in states:
            _, s, _ = state_metadata[state_id]
            if s == arb_state:
                occ += math.exp(forward_scores[state_id].item() - log_z_t)
    return occ


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--vocab-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--pos", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if args.seq_len < 2:
        raise SystemExit("seq-len must be >= 2 for the denominator spike")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    params = make_probs(args.vocab_size, args.frames, args.seed)
    seq = make_seq(args.vocab_size, args.seq_len, args.seed + 1)

    ll_self_ref = _ctc_forward(params, seq, blank=BLANK)
    ll_denom_ref, occ_ref = _ctc_forward_denom(params, seq, pos=args.pos, blank=BLANK)

    log_probs = torch.log(params.T.clamp(min=1e-30)).to(device=device, dtype=torch.float32)
    ctc_graph = k2.ctc_graph([seq.tolist()], modified=False, device=device)
    denom_graph = build_denom_graph(seq.tolist(), pos=args.pos, vocab_size=args.vocab_size, blank=BLANK)
    unrolled_ctc_graph = build_unrolled_ctc_graph(params, seq, blank=BLANK)
    unrolled_denom_graph, unrolled_denom_metadata, _ = build_unrolled_denom_graph_with_metadata(
        params,
        seq,
        pos=args.pos,
        blank=BLANK,
    )
    denom_topology = build_unrolled_denom_topology(
        seq,
        pos=args.pos,
        frames=params.shape[1],
        vocab_size=args.vocab_size,
        blank=BLANK,
    )

    ll_self_k2 = fsa_nll(log_probs, ctc_graph, device)
    ll_denom_k2 = fsa_nll(
        log_probs,
        k2.Fsa.from_str(denom_graph, acceptor=True),
        device,
    )
    ll_self_unrolled = weighted_fsa_nll(unrolled_ctc_graph)
    ll_denom_unrolled = weighted_fsa_nll(unrolled_denom_graph)
    occ_unrolled = unrolled_denom_occupancy_k2(
        unrolled_denom_graph,
        unrolled_denom_metadata,
        pos=args.pos,
        device=device,
    )
    topology_lattice = intersect_unrolled_denom_topology(
        denom_topology,
        log_probs,
        device=device,
    )
    ll_denom_topology = topology_denom_nll(topology_lattice)
    topology_forward = topology_lattice.get_forward_scores(
        use_double_scores=True,
        log_semiring=True,
    )
    occ_topology = denom_topology.occupancy_from_forward_scores(topology_forward)

    result = {
        "device": str(device),
        "seq": seq.tolist(),
        "pos": args.pos,
        "ll_self_ref": ll_self_ref,
        "ll_self_k2": ll_self_k2,
        "ll_self_abs_diff": abs(ll_self_ref - ll_self_k2),
        "ll_self_unrolled": ll_self_unrolled,
        "ll_self_unrolled_abs_diff": abs(ll_self_ref - ll_self_unrolled),
        "ll_denom_ref": ll_denom_ref,
        "ll_denom_k2": ll_denom_k2,
        "ll_denom_abs_diff": abs(ll_denom_ref - ll_denom_k2),
        "ll_denom_unrolled": ll_denom_unrolled,
        "ll_denom_unrolled_abs_diff": abs(ll_denom_ref - ll_denom_unrolled),
        "ll_denom_topology": ll_denom_topology,
        "ll_denom_topology_abs_diff": abs(ll_denom_ref - ll_denom_topology),
        "occ_ref": occ_ref,
        "occ_unrolled": occ_unrolled,
        "occ_unrolled_abs_diff": abs(occ_ref - occ_unrolled),
        "occ_topology": occ_topology,
        "occ_topology_abs_diff": abs(occ_ref - occ_topology),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    tol = 1e-5
    if not math.isfinite(ll_self_k2) or not math.isfinite(ll_denom_k2):
        return 2
    if result["ll_self_unrolled_abs_diff"] > tol:
        return 3
    if result["ll_denom_unrolled_abs_diff"] > tol:
        return 4
    if result["occ_unrolled_abs_diff"] > tol:
        return 5
    if result["ll_denom_topology_abs_diff"] > tol:
        return 6
    if result["occ_topology_abs_diff"] > tol:
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

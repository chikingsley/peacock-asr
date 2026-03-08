from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


BLANK = 0
_K2_MODULE: Any | None = None
logger = logging.getLogger("p001_gop")
_PRUNED_SEARCH_BEAM = 100.0
_OUTPUT_BEAM = 1_000_000.0
_MAX_ACTIVE_STATES = 10_000_000


def _require_k2() -> Any:
    global _K2_MODULE  # noqa: PLW0603
    if _K2_MODULE is None:
        _K2_MODULE = importlib.import_module("k2")
    return _K2_MODULE


@dataclass
class DenomTopology:
    graph_text: str
    final_state: int
    frame_state_ids: torch.Tensor
    frame_arb_mask: torch.Tensor
    dest_frame: torch.Tensor
    dest_logical_state: torch.Tensor
    _cpu_fsa: Any | None = field(default=None, init=False, repr=False)

    def to_fsa(self, device: torch.device | None = None) -> Any:
        k2 = _require_k2()
        if self._cpu_fsa is None:
            fsa = k2.Fsa.from_str(self.graph_text, acceptor=True)
            fsa.dest_frame = self.dest_frame.clone()
            fsa.dest_logical_state = self.dest_logical_state.clone()
            fsa, arc_map = k2.arc_sort(fsa, ret_arc_map=True)
            fsa.dest_frame = fsa.dest_frame[arc_map]
            fsa.dest_logical_state = fsa.dest_logical_state[arc_map]
            self._cpu_fsa = fsa
        if device is None:
            return self._cpu_fsa
        return self._cpu_fsa.to(device)

    def occupancy_from_forward_scores(
        self,
        forward_scores: torch.Tensor,
        *,
        state_offset: int = 0,
    ) -> float:
        state_ids = self.frame_state_ids.to(
            device=forward_scores.device,
            dtype=torch.long,
        )
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
        occ = probs.masked_select(
            self.frame_arb_mask.to(forward_scores.device) & valid_mask
        ).sum()
        return float(occ.item())


def k2_available() -> bool:
    try:
        _require_k2()
    except ModuleNotFoundError:
        return False
    return True


def _sort_arc_records(
    records: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    return sorted(records, key=lambda item: (item[0], item[2], item[1]))


def _topology_cache_dir() -> Path:
    from p001_gop.settings import settings  # noqa: PLC0415

    cache_dir = settings.cache_dir / "k2_topologies"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def topology_cache_dir() -> Path:
    """Return the persistent on-disk topology cache directory."""
    return _topology_cache_dir()


def _topology_cache_path(
    seq_key: tuple[int, ...],
    pos: int,
    frames: int,
    vocab_size: int,
    blank: int,
) -> Path:
    cache_key = f"{seq_key}|{pos}|{frames}|{vocab_size}|{blank}"
    digest = sha1(cache_key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _topology_cache_dir() / f"{digest}.pt"


def _load_cached_topology(path: Path) -> DenomTopology | None:
    if not path.exists():
        return None
    data = torch.load(path, weights_only=False)
    if not isinstance(data, dict):
        return None
    return DenomTopology(
        graph_text=str(data["graph_text"]),
        final_state=int(data["final_state"]),
        frame_state_ids=torch.as_tensor(data["frame_state_ids"], dtype=torch.int32),
        frame_arb_mask=torch.as_tensor(data["frame_arb_mask"], dtype=torch.bool),
        dest_frame=torch.as_tensor(data["dest_frame"], dtype=torch.int32),
        dest_logical_state=torch.as_tensor(
            data["dest_logical_state"],
            dtype=torch.int32,
        ),
    )


def _save_cached_topology(path: Path, topology: DenomTopology) -> None:
    payload = {
        "graph_text": topology.graph_text,
        "final_state": topology.final_state,
        "frame_state_ids": topology.frame_state_ids.cpu(),
        "frame_arb_mask": topology.frame_arb_mask.cpu(),
        "dest_frame": topology.dest_frame.cpu(),
        "dest_logical_state": topology.dest_logical_state.cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False, suffix=".pt") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _prune_topology_records(
    arcs: list[tuple[int, int, int]],
    *,
    final_state: int,
    state_metadata: dict[int, tuple[int, int, int]],
    final_frame: int,
    pos: int,
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
    dest_frames: list[int] = []
    dest_logical_states: list[int] = []
    kept_state_metadata: dict[int, tuple[int, int, int]] = {}

    for src, dst, label in _sort_arc_records(arcs):
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

    frame_state_ids, frame_arb_mask = _build_occupancy_layout(
        kept_state_metadata,
        pos=pos,
        frames=final_frame,
    )

    return DenomTopology(
        graph_text="\n".join([*kept_lines, f"{remap[final_state]}"]),
        final_state=remap[final_state],
        frame_state_ids=frame_state_ids,
        frame_arb_mask=frame_arb_mask,
        dest_frame=torch.tensor(dest_frames, dtype=torch.int32),
        dest_logical_state=torch.tensor(dest_logical_states, dtype=torch.int32),
    )


def _build_occupancy_layout(
    state_metadata: dict[int, tuple[int, int, int]],
    *,
    pos: int,
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    states_by_frame: dict[int, list[int]] = {}
    for state_id, (frame_idx, _, _) in state_metadata.items():
        states_by_frame.setdefault(frame_idx, []).append(state_id)

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


def _occupancies_from_pruned_lattice(
    lattice: Any,
    topologies: Sequence[DenomTopology],
) -> list[float]:
    forward_scores = lattice.get_forward_scores(
        use_double_scores=True,
        log_semiring=True,
    )
    source_state_ids = lattice.arcs.row_ids(2).to(dtype=torch.long)
    case_ids_per_state = lattice.arcs.row_ids(1).to(dtype=torch.long)
    case_ids_per_arc = case_ids_per_state[source_state_ids]
    arc_forward = forward_scores[source_state_ids] + lattice.scores.to(
        dtype=forward_scores.dtype
    )
    dest_frames = lattice.dest_frame.to(dtype=torch.long)
    dest_logical_states = lattice.dest_logical_state.to(dtype=torch.long)

    occupancies: list[float] = []
    for case_idx, topology in enumerate(topologies):
        case_mask = case_ids_per_arc == case_idx
        arb_state = 2 * case_idx + 1
        occ = 0.0
        frames = topology.frame_state_ids.shape[0]
        for frame_idx in range(frames):
            frame_mask = case_mask & (dest_frames == frame_idx)
            if not bool(frame_mask.any()):
                continue
            total_log = torch.logsumexp(arc_forward[frame_mask], dim=0)
            arb_mask = frame_mask & (dest_logical_states == arb_state)
            if bool(arb_mask.any()):
                arb_log = torch.logsumexp(arc_forward[arb_mask], dim=0)
                occ += float(torch.exp(arb_log - total_log).item())
        occupancies.append(occ)
    return occupancies


def _python_scalar_fns() -> tuple[Any, Any]:
    module = importlib.import_module("p001_gop.gop")
    return module._compute_scalar_terms_python, module._ctc_forward  # noqa: SLF001


@lru_cache(maxsize=8192)
def _build_unrolled_denom_topology_cached(
    seq_key: tuple[int, ...],
    pos: int,
    frames: int,
    vocab_size: int,
    blank: int,
) -> DenomTopology:
    cache_path = _topology_cache_path(seq_key, pos, frames, vocab_size, blank)
    cached = _load_cached_topology(cache_path)
    if cached is not None:
        return cached

    seq = tuple(int(x) for x in seq_key)
    seq_len = len(seq)
    big_l = 2 * seq_len + 1
    next_label = seq[pos + 1] if pos < seq_len - 1 else None

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
        return [token for token in range(vocab_size) if token not in exclude]

    def maybe_add_curr(
        curr_s: int,
        curr_p: int,
        preds: list[tuple[int, int]],
        label: int,
    ) -> None:
        valid_preds = [
            (prev_s, prev_p)
            for prev_s, prev_p in preds
            if (t - 1, prev_s, prev_p) in state_ids
        ]
        if not valid_preds:
            return
        curr = get_state(t, curr_s, curr_p)
        for prev_s, prev_p in valid_preds:
            arcs.append((get_state(t - 1, prev_s, prev_p), curr, label))

    arcs.append((0, get_state(0, 0, 0), blank))
    if pos == 0:
        excluded = {blank}
        if next_label is not None:
            excluded.add(next_label)
        arcs.extend(
            (0, get_state(0, 1, token), token)
            for token in allowed_arb_tokens(excluded)
        )
        if seq_len > 1:
            arcs.append((0, get_state(0, 3, 0), seq[1]))
    else:
        arcs.append((0, get_state(0, 1, 0), seq[0]))

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
                    excluded = {blank}
                    if next_label is not None:
                        excluded.add(next_label)
                    preds.extend(
                        (arb_idx, token)
                        for token in allowed_arb_tokens(excluded)
                    )
                elif s != 0:
                    preds.append((s - 1, 0))
                maybe_add_curr(s, 0, preds, blank)
                continue

            if pos != li and pos != li - 1:
                token_id = seq[li]
                preds = [(s, 0), (s - 1, 0)]
                if s != 1 and seq[li] != seq[li - 1]:
                    preds.append((s - 2, 0))
                maybe_add_curr(s, 0, preds, token_id)
                continue

            if pos == li - 1:
                token_id = seq[li]
                preds = [(s, 0), (s - 1, 0)]
                excluded = {blank, token_id}
                if next_label is not None:
                    excluded.add(next_label)
                preds.extend((s - 2, token) for token in allowed_arb_tokens(excluded))
                preds.append((s - 3, 0))
                if li - 2 >= 0 and seq[li - 2] != token_id:
                    preds.append((s - 4, 0))
                maybe_add_curr(s, 0, preds, token_id)
                continue

            excluded = {blank}
            if next_label is not None:
                excluded.add(next_label)
            for token in allowed_arb_tokens(excluded):
                preds: list[tuple[int, int]] = [(s, token), (s - 1, 0)]
                if s != 1 and token != seq[li - 1]:
                    preds.append((s - 2, 0))
                maybe_add_curr(s, token, preds, token)

    final_state = next_id
    for key, node_id in state_ids.items():
        if key[0] == frames - 1:
            arcs.append((node_id, final_state, -1))

    topology = _prune_topology_records(
        arcs,
        final_state=final_state,
        state_metadata=state_metadata,
        final_frame=frames,
        pos=pos,
    )
    _save_cached_topology(cache_path, topology)
    return topology


def prewarm_topology_cache(
    prepared: Sequence[tuple[np.ndarray, list[int], list[str], list[float]]],
    *,
    blank: int,
) -> dict[str, int]:
    """Populate the persistent topology cache for a prepared split."""
    total_cases = 0
    unique_topologies = 0
    cache_hits = 0
    cache_misses = 0
    seen: set[tuple[tuple[int, ...], int, int, int, int]] = set()

    for posteriors, phone_indices, _valid_phones, _valid_scores in prepared:
        seq_key = tuple(int(phone) for phone in phone_indices)
        frames = int(posteriors.shape[0])
        vocab_size = int(posteriors.shape[1])
        total_cases += len(seq_key)
        for pos in range(len(seq_key)):
            topo_key = (seq_key, pos, frames, vocab_size, blank)
            if topo_key in seen:
                continue
            seen.add(topo_key)
            unique_topologies += 1
            cache_path = _topology_cache_path(*topo_key)
            if cache_path.exists():
                cache_hits += 1
            else:
                cache_misses += 1
            _build_unrolled_denom_topology_cached(*topo_key)

    return {
        "prepared_utterances": len(prepared),
        "total_cases": total_cases,
        "unique_topologies": unique_topologies,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_files": len(list(_topology_cache_dir().glob("*.pt"))),
    }


def compute_scalar_terms_k2(
    posteriors: np.ndarray,
    phone_indices: Sequence[int],
    *,
    blank: int = BLANK,
    device: torch.device | None = None,
) -> tuple[float, list[float], list[float]]:
    _compute_scalar_terms_python, _ctc_forward = _python_scalar_fns()
    k2 = _require_k2()
    if device is None:
        device = torch.device("cpu")

    seq_key = tuple(int(phone) for phone in phone_indices)
    if not seq_key:
        return 0.0, [], []

    post_mat = torch.from_numpy(posteriors).double()
    params = post_mat.transpose(0, 1)
    seq = torch.tensor(seq_key, dtype=torch.int32)
    ll_self = _ctc_forward(params, seq, blank=blank)

    frames = posteriors.shape[0]
    vocab_size = posteriors.shape[1]
    topologies = [
        _build_unrolled_denom_topology_cached(
            seq_key,
            pos,
            frames,
            vocab_size,
            blank,
        )
        for pos in range(len(seq_key))
    ]
    fsas = [topology.to_fsa() for topology in topologies]
    fsa_vec = k2.Fsa.from_fsas(fsas).to(device)

    log_probs = torch.log(
        torch.from_numpy(posteriors).clamp(min=1e-30)
    ).to(device=device, dtype=torch.float32)
    log_probs_batch = (
        log_probs.unsqueeze(0).expand(len(topologies), -1, -1).contiguous()
    )
    supervisions = torch.tensor(
        [[case_idx, 0, frames] for case_idx in range(len(topologies))],
        dtype=torch.int32,
    )
    dense = k2.DenseFsaVec(log_probs_batch, supervisions)
    lattice = k2.intersect_dense_pruned(
        fsa_vec,
        dense,
        search_beam=_PRUNED_SEARCH_BEAM,
        output_beam=_OUTPUT_BEAM,
        min_active_states=0,
        max_active_states=_MAX_ACTIVE_STATES,
        allow_partial=False,
        seqframe_idx_name="seqframe",
        frame_idx_name="frame",
    )

    totals = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
    if not torch.isfinite(totals).all():
        logger.warning(
            "k2 scalar lattice was empty or non-finite; falling back to "
            "Python scalar path."
        )
        return _compute_scalar_terms_python(params, seq, blank)
    ll_denoms = [float(x) for x in (-totals).detach().cpu().tolist()]
    try:
        occupancies = _occupancies_from_pruned_lattice(lattice, topologies)
    except (IndexError, RuntimeError) as exc:
        logger.warning(
            "k2 scalar occupancy reconstruction failed (%s); falling back to "
            "Python scalar path.",
            exc,
        )
        return _compute_scalar_terms_python(params, seq, blank)
    scores = [-ll_self + ll_denom for ll_denom in ll_denoms]
    return ll_self, scores, occupancies


def compute_scalar_scores_k2(
    posteriors: np.ndarray,
    phone_indices: Sequence[int],
    *,
    blank: int = BLANK,
    device: torch.device | None = None,
) -> tuple[float, list[float]]:
    _compute_scalar_terms_python, _ctc_forward = _python_scalar_fns()
    k2 = _require_k2()
    if device is None:
        device = torch.device("cpu")

    seq_key = tuple(int(phone) for phone in phone_indices)
    if not seq_key:
        return 0.0, []

    post_mat = torch.from_numpy(posteriors).double()
    params = post_mat.transpose(0, 1)
    seq = torch.tensor(seq_key, dtype=torch.int32)
    ll_self = _ctc_forward(params, seq, blank=blank)

    frames = posteriors.shape[0]
    vocab_size = posteriors.shape[1]
    topologies = [
        _build_unrolled_denom_topology_cached(
            seq_key,
            pos,
            frames,
            vocab_size,
            blank,
        )
        for pos in range(len(seq_key))
    ]
    fsas = [topology.to_fsa() for topology in topologies]
    fsa_vec = k2.Fsa.from_fsas(fsas).to(device)

    log_probs = torch.log(
        torch.from_numpy(posteriors).clamp(min=1e-30)
    ).to(device=device, dtype=torch.float32)
    log_probs_batch = (
        log_probs.unsqueeze(0).expand(len(topologies), -1, -1).contiguous()
    )
    supervisions = torch.tensor(
        [[case_idx, 0, frames] for case_idx in range(len(topologies))],
        dtype=torch.int32,
    )
    dense = k2.DenseFsaVec(log_probs_batch, supervisions)
    lattice = k2.intersect_dense_pruned(
        fsa_vec,
        dense,
        search_beam=_PRUNED_SEARCH_BEAM,
        output_beam=_OUTPUT_BEAM,
        min_active_states=0,
        max_active_states=_MAX_ACTIVE_STATES,
        allow_partial=False,
    )
    totals = lattice.get_tot_scores(log_semiring=True, use_double_scores=True)
    if not torch.isfinite(totals).all():
        logger.warning(
            "k2 scalar score lattice was non-finite; falling back to Python "
            "scalar path."
        )
        ll_self_py, scores_py, _ = _compute_scalar_terms_python(params, seq, blank)
        return ll_self_py, scores_py

    ll_denoms = [float(x) for x in (-totals).detach().cpu().tolist()]
    scores = [-ll_self + ll_denom for ll_denom in ll_denoms]
    return ll_self, scores

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


EPS = "<eps>"


@dataclass(frozen=True)
class AlignmentResult:
    ref: list[str]
    hyp: list[str]
    ops: list[str]


def read_transcript_file(path: str | Path) -> dict[str, list[str]]:
    transcripts: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        utt_id = parts[0]
        phones = parts[1].split() if len(parts) > 1 else []
        transcripts[utt_id] = phones
    return transcripts


def align_tokens(ref: list[str], hyp: list[str]) -> AlignmentResult:
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    priority = {"C": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag_op = "C" if ref[i - 1] == hyp[j - 1] else "S"
            candidates = [
                (dp[i - 1][j - 1] + (0 if diag_op == "C" else 1), diag_op),
                (dp[i - 1][j] + 1, "D"),
                (dp[i][j - 1] + 1, "I"),
            ]
            best_cost, best_op = min(candidates, key=lambda item: (item[0], priority[item[1]]))
            dp[i][j] = best_cost
            back[i][j] = best_op

    i, j = n, m
    aligned_ref: list[str] = []
    aligned_hyp: list[str] = []
    ops: list[str] = []
    while i > 0 or j > 0:
        op = back[i][j]
        if op in {"C", "S"}:
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(hyp[j - 1])
            ops.append(op)
            i -= 1
            j -= 1
        elif op == "D":
            aligned_ref.append(ref[i - 1])
            aligned_hyp.append(EPS)
            ops.append("D")
            i -= 1
        elif op == "I":
            aligned_ref.append(EPS)
            aligned_hyp.append(hyp[j - 1])
            ops.append("I")
            j -= 1
        else:  # pragma: no cover - unreachable with valid DP backpointers
            raise RuntimeError("Alignment backtrace failed.")

    aligned_ref.reverse()
    aligned_hyp.reverse()
    ops.reverse()
    return AlignmentResult(ref=aligned_ref, hyp=aligned_hyp, ops=ops)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_mdd_metrics(
    human_seq: dict[str, list[str]],
    ref_seq: dict[str, list[str]],
    hyp_seq: dict[str, list[str]],
) -> dict[str, float]:
    utt_ids = sorted(set(hyp_seq) & set(human_seq) & set(ref_seq))

    cor_cor = cor_nocor = 0
    sub_sub = sub_sub1 = sub_nosub = 0
    ins_ins = ins_ins1 = ins_noins = 0
    del_del = del_del1 = del_nodel = 0

    for utt_id in utt_ids:
        ref_human = align_tokens(ref_seq[utt_id], human_seq[utt_id])
        human_our = align_tokens(human_seq[utt_id], hyp_seq[utt_id])
        ref_our = align_tokens(ref_seq[utt_id], hyp_seq[utt_id])

        flag = 0
        for idx in range(len(ref_human.ref)):
            if ref_human.ref[idx] == EPS:
                continue
            while flag < len(ref_our.ref) and ref_our.ref[flag] == EPS:
                flag += 1
            if flag >= len(ref_our.ref):
                break
            if ref_human.ref[idx] == ref_our.ref[flag] and ref_human.ref[idx] != EPS:
                if ref_human.ops[idx] == "D" and ref_our.ops[flag] == "D":
                    del_del += 1
                elif ref_human.ops[idx] == "D" and ref_our.ops[flag] not in {"D", "C"}:
                    del_del1 += 1
                elif ref_human.ops[idx] == "D" and ref_our.ops[flag] == "C":
                    del_nodel += 1
                flag += 1

        flag = 0
        for idx in range(len(ref_human.hyp)):
            if ref_human.hyp[idx] == EPS:
                continue
            while flag < len(human_our.ref) and human_our.ref[flag] == EPS:
                flag += 1
            if flag >= len(human_our.ref):
                break
            if ref_human.hyp[idx] == human_our.ref[flag] and ref_human.hyp[idx] != EPS:
                if ref_human.ops[idx] == "C" and human_our.ops[flag] == "C":
                    cor_cor += 1
                elif ref_human.ops[idx] == "C" and human_our.ops[flag] != "C":
                    cor_nocor += 1

                if ref_human.ops[idx] == "S" and human_our.ops[flag] == "C":
                    sub_sub += 1
                elif ref_human.ops[idx] == "S" and human_our.ops[flag] != "C" and ref_human.ref[idx] != human_our.hyp[flag]:
                    sub_sub1 += 1
                elif ref_human.ops[idx] == "S" and human_our.ops[flag] != "C" and ref_human.ref[idx] == human_our.hyp[flag]:
                    sub_nosub += 1

                if ref_human.ops[idx] == "I" and human_our.ops[flag] == "C":
                    ins_ins += 1
                elif ref_human.ops[idx] == "I" and human_our.ops[flag] not in {"C", "D"}:
                    ins_ins1 += 1
                elif ref_human.ops[idx] == "I" and human_our.ops[flag] == "D":
                    ins_noins += 1
                flag += 1

    tp = sub_sub + ins_ins + del_del + sub_sub1 + ins_ins1 + del_del1
    fp = cor_nocor
    fn = sub_nosub + ins_noins + del_nodel
    err_count = sub_sub + sub_sub1 + sub_nosub + ins_ins + ins_ins1 + ins_noins + del_del + del_del1 + del_nodel
    false_accept = sub_nosub + ins_noins + del_nodel
    correct_diag = sub_sub + ins_ins + del_del
    error_diag = sub_sub1 + ins_ins1 + del_del1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "TA": _safe_div(cor_cor, cor_cor + cor_nocor),
        "FR": _safe_div(cor_nocor, cor_cor + cor_nocor),
        "FA": _safe_div(false_accept, err_count),
        "Correct Diag": _safe_div(correct_diag, correct_diag + error_diag),
        "Error Diag": _safe_div(error_diag, correct_diag + error_diag),
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "FAR": 1 - recall,
        "FRR": _safe_div(cor_nocor, cor_nocor + cor_cor),
        "DER": _safe_div(error_diag, error_diag + correct_diag),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
    }


def write_result(path: str | Path, metrics: dict[str, float]) -> None:
    ordered_keys = ["TA", "FR", "FA", "Correct Diag", "Error Diag", "Recall", "Precision", "F1", "FAR", "FRR", "DER"]
    lines = [f"{key}: {metrics[key]:.4f}" for key in ordered_keys]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--human-seq", type=str, default=None, help="Path to realized-phone transcript file.")
    parser.add_argument("--ref", type=str, default=None, help="Path to canonical-phone transcript file.")
    parser.add_argument("--hyp", type=str, default=None, help="Path to hypothesis transcript file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output text file path.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text summary.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    exp_dir = Path(args.exp_dir)
    human_seq_path = Path(args.human_seq) if args.human_seq else exp_dir / "rel_nosil"
    ref_path = Path(args.ref) if args.ref else exp_dir / "can_nosil"
    hyp_path = Path(args.hyp) if args.hyp else exp_dir / "hyp_nosil"

    if not human_seq_path.exists() or not ref_path.exists() or not hyp_path.exists():
        raise FileNotFoundError(
            "Expected transcript files were not found. Run p012-recog with --remove-sil --remove-special-token first."
        )

    metrics = compute_mdd_metrics(
        read_transcript_file(human_seq_path),
        read_transcript_file(ref_path),
        read_transcript_file(hyp_path),
    )
    if args.output:
        write_result(args.output, metrics)

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        ordered_keys = ["TA", "FR", "FA", "Correct Diag", "Error Diag", "Recall", "Precision", "F1", "FAR", "FRR", "DER"]
        for key in ordered_keys:
            print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()

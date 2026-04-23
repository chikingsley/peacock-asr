#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${project_dir}/../.." && pwd)"

run_root="${1:-${project_dir}/runs/repro-20260403-094834}"
orig_eval="${repo_root}/projects/P014-hmamba-original/eval_mdd"
kaldi_bin="${project_dir}/third_party/kaldi/src/bin"

if [[ ! -d "${run_root}" ]]; then
  echo "Run root not found: ${run_root}" >&2
  exit 1
fi

run_root="$(cd "${run_root}" && pwd)"

if [[ ! -x "${kaldi_bin}/align-text" || ! -x "${kaldi_bin}/compute-wer" ]]; then
  echo "Kaldi binaries not found under ${kaldi_bin}" >&2
  exit 1
fi

if [[ ! -d "${orig_eval}" ]]; then
  echo "Original HMamba eval_mdd directory not found: ${orig_eval}" >&2
  exit 1
fi

seed_count=0
summary_lines=()

for seed_dir in "${run_root}"/seed*; do
  [[ -d "${seed_dir}" ]] || continue

  rel="${seed_dir}/rel_nosil"
  can="${seed_dir}/can_nosil"
  hyp="${seed_dir}/hyp_nosil"
  for required in "${rel}" "${can}" "${hyp}"; do
    if [[ ! -f "${required}" ]]; then
      echo "Missing required file for $(basename "${seed_dir}"): ${required}" >&2
      exit 1
    fi
  done

  work="${seed_dir}/kaldi_mdd_eval"
  raw_out="${seed_dir}/mdd_result_kaldi_raw.txt"
  parsed_out="${seed_dir}/mdd_result_kaldi.txt"

  rm -rf "${work}"
  mkdir -p "${work}"
  ln -s "${orig_eval}" "${work}/eval_mdd"

  (
    cd "${work}"
    PATH="${kaldi_bin}:${PATH}" bash eval_mdd/mdd_result.sh "${rel}" "${can}" "${hyp}"
  ) | tee "${raw_out}" >/dev/null

  awk '
    /^TA:/ {print "ta=" $2}
    /^FR:/ {print "fr=" $2}
    /^FA:/ {print "fa=" $2}
    /^Correct Diag:/ {print "correct_diag=" $3}
    /^Error Diag:/ {print "error_diag=" $3}
    /^Recall:/ {print "recall=" $2}
    /^Precision:/ {print "precision=" $2}
    /^F1:/ {print "f1=" $2}
    /^FAR:/ {print "far=" $2}
    /^FRR:/ {print "frr=" $2}
    /^DER:/ {print "der=" $2}
    /^%WER/ {print "wer=" $2}
  ' "${raw_out}" > "${parsed_out}"

  precision="$(grep '^precision=' "${parsed_out}" | cut -d= -f2)"
  recall="$(grep '^recall=' "${parsed_out}" | cut -d= -f2)"
  f1="$(grep '^f1=' "${parsed_out}" | cut -d= -f2)"
  wer="$(grep '^wer=' "${parsed_out}" | cut -d= -f2)"

  summary_lines+=("$(basename "${seed_dir}") ${precision} ${recall} ${f1} ${wer}")
  seed_count=$((seed_count + 1))
done

if [[ "${seed_count}" -eq 0 ]]; then
  echo "No seed directories found under ${run_root}" >&2
  exit 1
fi

printf '%s\n' "${summary_lines[@]}" | sort

#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mac_github_dir="${MAC_GITHUB_DIR:-/Users/simonpeacocks/GitHub}"
fluid_dir="${FLUIDAUDIO_DIR:-$mac_github_dir/FluidAudio}"
limit="${MOSS_BENCH_LIMIT:-2620}"
page_size="${MOSS_BENCH_PAGE_SIZE:-100}"
summary_every="${MOSS_BENCH_SUMMARY_EVERY:-25}"

mkdir -p "$project_dir/artifacts/logs" "$project_dir/artifacts/evals"

run_parakeet_v3() {
  if [[ ! -d "$fluid_dir" ]]; then
    echo "FluidAudio checkout not found at $fluid_dir; skipping Parakeet v3 benchmark"
    return 0
  fi

  local output_dir="$project_dir/artifacts/evals/fluid-parakeet-v3-librispeech-test-clean-full"
  mkdir -p "$output_dir"

  echo "== Parakeet TDT v3 / FluidAudio / LibriSpeech test-clean =="
  (
    cd "$fluid_dir"
    swift build -c release
    .build/release/fluidaudiocli asr-benchmark \
      --subset test-clean \
      --model-version v3 \
      --output "$output_dir/results.json"
  )
}

run_moss_eval() {
  local name="$1"
  local model_dir="$2"
  local output_dir="$project_dir/artifacts/evals/librispeech-test-clean-${name}-full"

  echo "== MOSS $name / LibriSpeech test-clean =="
  uv run --directory "$project_dir" --extra mac --locked moss-streaming-eval \
    --model-dir "$model_dir" \
    --limit "$limit" \
    --page-size "$page_size" \
    --output-dir "$output_dir" \
    --resume \
    --quiet \
    --summary-every "$summary_every"
}

run_moss_comparison() {
  local right_name="$1"
  local right_dir="$2"
  local output_dir="$project_dir/artifacts/evals/librispeech-test-clean-mlx-bf16-vs-${right_name}-full"
  local bf16_jsonl="$project_dir/artifacts/evals/librispeech-test-clean-mlx-bf16-full/predictions.jsonl"
  local right_jsonl="$right_dir/predictions.jsonl"

  if [[ ! -s "$bf16_jsonl" || ! -s "$right_jsonl" ]]; then
    echo "Skipping comparison for $right_name; missing prediction JSONL"
    return 0
  fi

  uv run --directory "$project_dir" --extra mac --locked moss-compare-evals \
    --left "$bf16_jsonl" \
    --right "$right_jsonl" \
    --left-name mlx-bf16 \
    --right-name "$right_name" \
    --output-dir "$output_dir"
}

run_parakeet_v3

run_moss_eval \
  "mlx-bf16" \
  "$project_dir/artifacts/mlx/MOSS-Transcribe-preview-2B-bf16"

run_moss_eval \
  "mlx-text-decoder-4bit-g64" \
  "$project_dir/artifacts/mlx/MOSS-Transcribe-preview-2B-text-decoder-4bit-g64"

run_moss_eval \
  "mlx-all-8bit-g64" \
  "$project_dir/artifacts/mlx/MOSS-Transcribe-preview-2B-all-8bit-g64"

run_moss_eval \
  "mlx-all-4bit-g64" \
  "$project_dir/artifacts/mlx/MOSS-Transcribe-preview-2B-all-4bit-g64"

run_moss_eval \
  "mlx-text-decoder-8bit-g64" \
  "$project_dir/artifacts/mlx/MOSS-Transcribe-preview-2B-text-decoder-8bit-g64"

run_moss_comparison \
  "mlx-text-decoder-4bit-g64" \
  "$project_dir/artifacts/evals/librispeech-test-clean-mlx-text-decoder-4bit-g64-full"

run_moss_comparison \
  "mlx-all-8bit-g64" \
  "$project_dir/artifacts/evals/librispeech-test-clean-mlx-all-8bit-g64-full"

run_moss_comparison \
  "mlx-all-4bit-g64" \
  "$project_dir/artifacts/evals/librispeech-test-clean-mlx-all-4bit-g64-full"

run_moss_comparison \
  "mlx-text-decoder-8bit-g64" \
  "$project_dir/artifacts/evals/librispeech-test-clean-mlx-text-decoder-8bit-g64-full"

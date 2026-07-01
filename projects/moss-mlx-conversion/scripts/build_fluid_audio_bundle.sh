#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
coreml_build_dir="${MOSS_COREML_BUILD_DIR:-$project_dir/coreml/build}"
bundle_dir="${MOSS_BUNDLE_DIR:-$project_dir/bundles/moss-fluid-audio-coreml-active}"
copy_mode="${MOSS_COPY_MODE:-clone}"
include_matched_768="${MOSS_INCLUDE_MATCHED_768:-0}"
overwrite="${MOSS_BUNDLE_OVERWRITE:-0}"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required bundle source: $path" >&2
    exit 1
  fi
}

copy_item() {
  local src="$1"
  local dest="$2"
  require_path "$src"
  mkdir -p "$(dirname "$dest")"
  case "$copy_mode" in
    symlink)
      ln -s "$src" "$dest"
      ;;
    copy)
      cp -R "$src" "$dest"
      ;;
    clone)
      if [[ "$(uname -s)" == "Darwin" ]]; then
        cp -cR "$src" "$dest"
      else
        cp -a --reflink=auto "$src" "$dest"
      fi
      ;;
    *)
      echo "Unknown MOSS_COPY_MODE=$copy_mode; use clone, copy, or symlink" >&2
      exit 1
      ;;
  esac
}

write_manifest() {
  local manifest="$bundle_dir/moss_bundle_manifest.json"
  if [[ "$include_matched_768" == "1" ]]; then
    cat >"$manifest" <<'JSON'
{
  "version": 1,
  "default_cache_preset": "compat-768",
  "artifacts": {
    "token_package_path": "compiled/moss_token_embedding.mlmodelc",
    "audio_package_path": "compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
    "prefill_cache_package_path": "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
    "step_package_path": "compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
    "tokenizer_path": "moss_tokenizer.json",
    "runtime_manifest_path": "moss_runtime_manifest.json"
  },
  "cache_presets": [
    {
      "name": "short-512",
      "aliases": ["cache-512", "512"],
      "description": "512-token padded prefill plus 512-cache decoder step for short prompt+decode windows.",
      "status": "validated-20-row-short-gate",
      "prefill_cache_package_path": "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
      "prefill_cache_seq_len": 512,
      "step_package_path": "compiled_step_padded_512/moss_decoder_step_padded_512.mlmodelc",
      "cache_len": 512,
      "max_total_positions": 512
    },
    {
      "name": "compat-768",
      "aliases": ["cache-768", "768"],
      "description": "512-token padded prefill plus 768-cache decoder step compatibility path.",
      "status": "validated-20-row-compat-gate",
      "prefill_cache_package_path": "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
      "prefill_cache_seq_len": 512,
      "step_package_path": "compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
      "cache_len": 768,
      "max_total_positions": 768
    },
    {
      "name": "matched-768",
      "description": "768-token padded prefill plus 768-cache decoder step. Torch-validates but currently fails in CoreML/MPSGraph cpu-gpu execution.",
      "status": "experimental-mpsgraph-blocked",
      "prefill_cache_package_path": "compiled_prefill_cache_768/moss_decoder_prefill_cache_768.mlmodelc",
      "prefill_cache_seq_len": 768,
      "step_package_path": "compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
      "cache_len": 768,
      "max_total_positions": 768
    }
  ]
}
JSON
  else
    cat >"$manifest" <<'JSON'
{
  "version": 1,
  "default_cache_preset": "compat-768",
  "artifacts": {
    "token_package_path": "compiled/moss_token_embedding.mlmodelc",
    "audio_package_path": "compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
    "prefill_cache_package_path": "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
    "step_package_path": "compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
    "tokenizer_path": "moss_tokenizer.json",
    "runtime_manifest_path": "moss_runtime_manifest.json"
  },
  "cache_presets": [
    {
      "name": "short-512",
      "aliases": ["cache-512", "512"],
      "description": "512-token padded prefill plus 512-cache decoder step for short prompt+decode windows.",
      "status": "validated-20-row-short-gate",
      "prefill_cache_package_path": "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
      "prefill_cache_seq_len": 512,
      "step_package_path": "compiled_step_padded_512/moss_decoder_step_padded_512.mlmodelc",
      "cache_len": 512,
      "max_total_positions": 512
    },
    {
      "name": "compat-768",
      "aliases": ["cache-768", "768"],
      "description": "512-token padded prefill plus 768-cache decoder step compatibility path.",
      "status": "validated-20-row-compat-gate",
      "prefill_cache_package_path": "compiled_prefill_cache_512/moss_decoder_prefill_cache_512.mlmodelc",
      "prefill_cache_seq_len": 512,
      "step_package_path": "compiled_step_padded/moss_decoder_step_padded_fixture.mlmodelc",
      "cache_len": 768,
      "max_total_positions": 768
    }
  ]
}
JSON
  fi
}

if [[ -e "$bundle_dir" ]]; then
  if [[ "$overwrite" != "1" ]]; then
    echo "Bundle destination already exists: $bundle_dir" >&2
    echo "Set MOSS_BUNDLE_OVERWRITE=1 to replace it." >&2
    exit 1
  fi
  rm -rf "$bundle_dir"
fi

mkdir -p "$bundle_dir"

copy_item "$coreml_build_dir/compiled" "$bundle_dir/compiled"
copy_item "$coreml_build_dir/compiled_audio_30s" "$bundle_dir/compiled_audio_30s"
copy_item "$coreml_build_dir/compiled_prefill_cache_512" "$bundle_dir/compiled_prefill_cache_512"
copy_item "$coreml_build_dir/compiled_step_padded" "$bundle_dir/compiled_step_padded"
copy_item "$coreml_build_dir/compiled_step_padded_512" "$bundle_dir/compiled_step_padded_512"

if [[ "$include_matched_768" == "1" ]]; then
  copy_item "$coreml_build_dir/compiled_prefill_cache_768" "$bundle_dir/compiled_prefill_cache_768"
fi

copy_item "$project_dir/artifacts/coreml/moss_tokenizer.json" "$bundle_dir/moss_tokenizer.json"
copy_item "$project_dir/runtime/moss_runtime_manifest.json" "$bundle_dir/moss_runtime_manifest.json"
write_manifest

echo "Built MOSS FluidAudio bundle: $bundle_dir"
du -sh "$bundle_dir"

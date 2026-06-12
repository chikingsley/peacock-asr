#!/usr/bin/env bash
# Post-Georgian-training GPU queue: best-checkpoint eval, then the raw omniASR-LLM
# ceiling bench on all three languages (results comparable to per-project EXPERIMENTS).
set -x
cd "$(dirname "$0")/.." || exit 1

echo "=== 1/4 georgian eval: base vs v0_step_29000 ==="
uv run --project projects/georgian-asr georgian-eval \
  --models base=omni_ctc_300m_v2_georgian_base v0=omni_ctc_300m_v2_georgian_v0_step_29000

echo "=== 2/4 LLM bench: tajik (fleurs + conversational held-out) ==="
uv run --project projects/tajik-asr python scripts/bench_omni_llm.py \
  projects/tajik-asr/data/datasets/v3/version=0 tgk_Cyrl \
  --corpus-prefix fleurs --corpus-prefix youtube-

echo "=== 3/4 LLM bench: georgian (v0 test) ==="
uv run --project projects/tajik-asr python scripts/bench_omni_llm.py \
  projects/georgian-asr/data/datasets/v0/version=0 kat_Geor

echo "=== 4/4 LLM bench: persian (FLEURS dev, staged) ==="
uv run --project projects/tajik-asr python scripts/bench_omni_llm.py \
  projects/persian-asr/data/eval-v4/version=0 fas_Arab

echo "=== GPU QUEUE COMPLETE ==="

# Track 05 Ablation Plan

## What "Ablation" Means

Ablation = a controlled experiment where we change one component and keep all other factors fixed, so performance changes can be attributed to that one change.

For this track, examples:
- Change only the scorer (`poly` vs `SVR` vs `GOPT`), keep GOP method/data/splits fixed.
- Change only GOP method (`posterior` vs `logit` vs `constrained`), keep scorer/data/splits fixed.

## Goal

Identify which factor contributes most to PCC on SpeechOcean762:
- GOP scoring variant
- Downstream scorer
- Their interaction

## Frozen Setup (must stay constant)

- Dataset and split protocol
- Evaluation metric computation
- Feature extraction schema (except where GOP variant explicitly changes it)
- Training seeds policy

## Primary Metrics

- `PCC` (primary)
- `95% CI` for PCC (bootstrap or fixed repo method)
- Optional: per-phone diagnostics

## Minimal Experiment Matrix (Phase 1)

Use one backend first (current strongest/reliable) to isolate effects.

| Run ID | GOP Variant | Scorer | Purpose |
|---|---|---|---|
| A1 | Posterior GOP-SF (current) | Poly | Baseline anchor |
| A2 | Posterior GOP-SF (current) | SVR | Scorer ablation |
| A3 | Posterior GOP-SF (current) | GOPT | Scorer ablation |
| B1 | Logit GOP-SF (new) | Poly | Algorithmic effect |
| B2 | Logit GOP-SF (new) | SVR | Algorithmic + scorer interaction |
| B3 | Logit GOP-SF (new) | GOPT | Algorithmic + scorer interaction |

Phase-1 status (2026-03-03):
- Completed baseline trio with frozen backend `xlsr-espeak`.
- Results folder: `runs/2026-03-03_001037_track05_phase1_baseline/`.
- Observed means: `A1 PCC 0.3195`, `A2 PCC 0.5747`, `A3 PCC 0.6774 ± 0.0127` (5 seeds).

## Phase 2 (if Logit is Promising)

Add constrained substitutions to the better of `posterior`/`logit`:

| Run ID | GOP Variant | Scorer | Purpose |
|---|---|---|---|
| C1 | Constrained GOP-SF | Best scorer from A1-A3 | Test precision-oriented variant |
| C2 | Constrained GOP-SF | Second-best scorer | Check robustness |

## Decision Rules

- `Adopt new variant` if PCC improves by a practically meaningful margin (suggested `>= 0.015`) and CI trend is consistent.
- `Keep as optional` if gains are small/inconsistent but diagnostics improve.
- `Reject` if no meaningful gain and added complexity is high.

## Deliverables to Capture per Run

- Config snapshot (model, scorer, GOP flags)
- Exact command
- Output metrics JSON/log path
- One-line summary for results table

## Paper Mapping

- Methods: define GOP variants and scorer heads.
- Results: table A/B/C runs + CI.
- Discussion: why specific variants help/hurt.

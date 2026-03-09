# Agent Runbook

Use this project when delegating language-by-language dataset research.

## Goal

For one language at a time, produce a clean inventory of:

- public/open speech datasets usable for ASR
- licensed/commercial datasets
- approximate hours
- transcript quality and caveats
- whether the dataset was already used in a released NVIDIA FastConformer model

## Required Outputs

For each language agent:

1. Update the language note under `docs/languages/<lang>.md`
2. Write draft rows to `inventories/drafts/<lang>.tsv`
3. Leave unresolved questions in the language note under `Open Questions`

Do not edit `inventories/seed_datasets.tsv` or
`inventories/vendor_sources.tsv` in parallel with other agents.

Those shared files are for consolidation after the language passes finish.

## Research Rules

- Prefer official dataset pages, official Hugging Face cards, or official
  project repos.
- Distinguish clearly between:
  - `hours_used_in_nvidia_recipe`
  - `estimated_total_public_hours`
- Do not silently convert licensed data into "public."
- If license status is unclear, mark it `needs_audit`.
- Do not guess punctuation restoration, speaker counts, or transcript fidelity.
- Do not download raw corpora by default.
- Do not upload datasets to Hugging Face by default.
- Do not mirror licensed or access-restricted data anywhere.

## Minimum Acceptance For A Language Pass

- at least the NVIDIA seed recipe datasets are captured
- public vs licensed status is marked
- obvious major open corpora for the language are either added or explicitly
  marked missing
- one paragraph states whether a public-only reproduction is realistic

## Draft TSV Format

Each language draft file should use this header:

```text
language	dataset_name	hours_used_in_nvidia_recipe	estimated_total_public_hours	access_class	role	source_url	notes
```

The shared `inventories/seed_datasets.tsv` file uses the same header so draft
rows can be consolidated without losing the estimated public-hours column.

## Current Seed Sources

- NVIDIA FastConformer model cards
- NeMo training scripts/configs
- LDC catalog
- Appen OTS/custom collection pages
- DataTang dataset marketplace pages

# P009 Multilingual Data Map

Seed workspace for building language-specific ASR data inventories around the
FastConformer / Parakeet line.

The immediate goal is not training. The immediate goal is to answer, per
language:

- what public data exists
- what NVIDIA already used in their released FastConformer recipes
- what is open vs licensed vs commercial
- what is realistic to reproduce now
- what would be needed to scale toward a "Parakeet-like" model later

Current seeded languages:

- Russian
- Spanish
- Italian
- French

## Parallel Agent Workflow

Use this project in two phases.

### Phase 1: inventory only

One agent per language.

Each language agent should only touch:

- `docs/languages/<lang>.md`
- `inventories/drafts/<lang>.tsv`

They should **not** edit the shared inventory files in parallel.

### Phase 2: consolidation

After the language passes are done, one consolidator agent or one human merges
the draft TSVs into:

- `inventories/seed_datasets.tsv`
- `inventories/vendor_sources.tsv`

## What Agents Should Not Do By Default

- do not download raw audio corpora by default
- do not upload corpora to Hugging Face by default
- do not mirror licensed datasets anywhere
- do not start data cleaning pipelines by default

The default job is research and inventory, not acquisition.

If a later phase is approved, agents can:

- download metadata or manifests
- mirror public/open manifests to Hugging Face
- prepare acquisition scripts for public datasets

Raw audio upload should be a separate explicit step because license and storage
rules vary by dataset.

## Structure

- `docs/README.md`: project doc index
- `docs/LANGUAGE_MATRIX.md`: cross-language summary
- `docs/AGENT_RUNBOOK.md`: instructions for research agents
- `docs/languages/`: per-language inventory notes
- `docs/languages/*_dossier.md`: deeper evidence dossiers for high-priority
  languages
- `docs/agent_briefs/`: one brief per language for delegated research
- `inventories/drafts/`: per-language machine-readable draft inventories
- `inventories/seed_datasets.tsv`: machine-readable seed inventory
- `inventories/vendor_sources.tsv`: commercial / licensed source registry
- `scripts/validate_inventories.py`: TSV/header/consistency validator for draft
  and shared inventories

## Scope

This project is the staging area for multilingual data strategy, not a runtime
or modeling package.

When a language is ready for actual training, it should graduate into its own
project or into the owning ASR project.

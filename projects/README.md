# Projects Workspace

This directory is the canonical project-level workspace for ASR research.

## Naming

- Canonical IDs use `P###-slug`.
- Legacy `trackXX_*` names remain valid historical labels, but they are no longer the primary naming system.
- Every experiment config, run group, and research artifact should map to exactly one project ID.

## Layout (per project)

Each project workspace uses the same shape:

- `code/` - project-specific code and wrappers
- `docs/` - project notes, runbooks, decisions
- `papers/` - paper PDFs and reading notes
- `experiments/` - sweep configs, run manifests, outputs metadata
- `third_party/` - upstream/reference repos for this project
- `env/` - project-specific environment files

## Rules

- Keep reusable shared code in `src/peacock_asr/`.
- Keep project-specific logic in `projects/P###-.../code/`.
- Keep root-level `runs/` and `docs/research/track*` paths as compatibility paths during migration.
- Do not mix unrelated papers/references across project directories.

## Migration Policy

- Migrate in small steps by project.
- Keep legacy path mappings in `projects/INDEX.yaml` until each project is fully cut over.

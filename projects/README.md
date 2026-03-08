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
- `.env.example` - checked-in template for local secrets and project defaults
- `scripts/` - project-local utility scripts, including remote env sync helpers

## Rules

- Keep runtime code inside the owning project workspace, even when it means duplication.
- Keep project-specific logic in `projects/P###-.../code/`.
- Give each active project its own `pyproject.toml`, `.venv`, tests, and W&B command surface.
- Keep `docs/research/archived/0X_*.md` narratives as historical notes.
- Do not reintroduce root-level experiment directories like `runs/`.
- Do not mix unrelated papers/references across project directories.

## Migration Policy

- Migrate in small steps by project.
- Keep legacy path mappings in `projects/INDEX.yaml` until each project is fully cut over.
- Remove root runtime/package surfaces once the owning projects pass independently.

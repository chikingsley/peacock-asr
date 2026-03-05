# Papers Naming Convention

Scope: `docs/papers/` (shared literature mirror for all projects)

Status: Canonical naming policy for the flattened layout.

## 1. Goals

1. Keep every paper easy to find by filename.
2. Keep a single shared copy of each paper artifact.
3. Avoid folder sprawl by storing paper files in one flat directory.
4. Track categories as metadata, not directory paths.

## 2. Directory Layout (Flat)

All paper artifacts live directly in `docs/papers/`:

- `*.pdf` paper files
- `*.md` extracted notes/summaries
- `*.txt` extracted text (optional)

Category folders are not used for active storage.

## 3. Category Tracking (Pointers)

Categories are tracked via metadata files:

1. `docs/papers/CATEGORY_POINTERS.tsv`:
   - maps filename -> category
   - captures redistribution status and legacy category
2. `docs/papers/PAPER_INDEX.tsv` (recommended):
   - one row per unique paper
   - stores canonical metadata (author, year, DOI/arXiv, linked projects)

## 4. Filename Rules

### 4.1 Preferred format

`<arxiv_or_year>__[<Author et al, Year>]-<title-slug>.<ext>`

or the shorter already-used style:

`[Author et al, Year]-<title-slug>.<ext>`

Use one style consistently for new additions.

### 4.2 Sidecar alignment

For a given paper, keep the same basename for `.pdf`, `.md`, `.txt`.

### 4.3 Character rules

1. ASCII only.
2. Lowercase slug section.
3. Use `-` between slug words.
4. Keep slug concise and meaningful.

## 5. Duplicate/Collision Rules

1. If a new file collides with an existing basename, prefer one canonical file.
2. Keep alternates only when necessary, with explicit suffixes like:
   - `__from-<legacy-category>`
   - `__dupN`
3. Record any non-canonical duplicates in `CATEGORY_POINTERS.tsv` and/or `PAPER_INDEX.tsv` notes.

## 6. Project Linking Rules

1. Projects reference shared paper files under `docs/papers/`.
2. Do not duplicate PDFs into project folders unless absolutely required.
3. If project-local copies are needed, record the reason in project docs.

## 7. Migration Rules

1. New imports must follow this convention.
2. Legacy folder paths should be flattened to `docs/papers/<filename>`.
3. After moving files, update references (`.bib`, markdown links, JSON indices).
4. Validate:
   - no broken markdown links
   - all `.bib file={...}` paths resolve

## 8. Current Metadata Files

- `docs/papers/REDISTRIBUTION_MAP.tsv` (migration record)
- `docs/papers/CATEGORY_POINTERS.tsv` (category pointers)

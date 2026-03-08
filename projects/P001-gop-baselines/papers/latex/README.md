# P001 LaTeX Draft

This is the working LaTeX package for `P001`.

It is intentionally venue-neutral for now. The goal is to give the next writing
pass a real paper target instead of continuing to overload the markdown
manuscript with both notes and prose.

Current sources:

- narrative scaffold: `../../docs/manuscript.md`
- bibliography: `../../docs/refs.bib`
- final campaign outputs: `../../experiments/final/results/`

Build locally with:

```bash
latexmk -pdf main.tex
```

The next writing pass should fill each section deliberately rather than trying
to auto-convert the markdown draft wholesale.

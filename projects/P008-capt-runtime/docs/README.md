# P008 Docs

This project is the product/runtime layer that sits on top of the existing
pronunciation assessment work.

Core docs:

- `ARCHITECTURE.md` — four-box runtime architecture and boundaries
- `CANONICALIZER.md` — dictionary store, phone representation, and fallback logic
- `G2P_BAKEOFF.md` — candidate matrix and evaluation protocol

Project relationship:

- `P001` remains the main pronunciation scorer (`GOP-SF + GOPT`)
- `P002` remains the advanced scorer track
- `P008` owns runtime composition, multilingual canonicalization, alignment,
  and feedback assembly

# Advanced Scorers

This directory owns scorer families that require a richer data or feature
contract than the fixed `P001` phone-level feature pipeline.

Scope:

- faithful HMamba reproduction
- HiPAMA reproduction
- other scorer families that require phone/word/utterance structure,
  multi-aspect targets, or expanded supervision beyond the plain `P001`
  `UtteranceFeats` contract

Non-scope:

- simple scorer swaps on the existing `P001` feature contract
- the exploratory phone-level HMamba adaptation already implemented in
  `projects/P001-gop-baselines`

Why this lives in `P002`:

- `P002` is the richer-scoring / richer-contract lane
- these models are not just alternative heads on the same frozen `P001`
  tensors; they want additional structure and should be compared together


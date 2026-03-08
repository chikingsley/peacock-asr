# Scorer Families in P002

This note clarifies what a "scorer" is in this repo and why some papers belong
in `P001` while others belong in `P002`.

## 1. What changes when you change the scorer?

In our pronunciation pipeline there are three separable layers:

1. **Posterior generator**
   - acoustic / phoneme recognizer
   - emits phone posteriors or logits over time
2. **Intermediate pronunciation representation**
   - GOP scores
   - GOP-derived phone features
   - optional alignment, duration, energy, SSL embeddings
3. **Scoring head**
   - maps the intermediate representation to human labels

Changing the scorer is meaningful when the score target is human-rated phone,
word, or utterance quality. The question is not "can we overfit the benchmark?"
The question is:

- what inductive bias best maps pronunciation evidence to the human labels?

That can be legitimate modeling, but only if the contract is clear.

## 2. P001 vs P002 boundary

### P001: fixed feature contract, simple scorer swaps

`P001` owns:

- scalar GOP
- feature vector + SVR
- feature vector + GOPT
- exploratory phone-level HMamba adaptation

These all consume the same basic phone-level GOP feature contract.

### P002: richer feature/data contract

`P002` owns methods that require more than the frozen `P001` phone-level
contract:

- ConPCO / HierTFR / HierCB line
- faithful HMamba
- HiPAMA
- other multi-aspect / hierarchical scorer families

The core reason is structural:

- richer features
- phone/word/utterance hierarchy
- multi-aspect targets
- extra alignment or text/audio interaction

## 3. Family-by-family read

### GOPT

What it changes:

- keeps GOP features
- replaces simple regression with a contextual phone-level transformer

Why it matters:

- nearby phones influence each other
- pronunciation quality is not independent phone-by-phone

What we learned:

- in `P001`, this matters a lot

### ConPCO

What it changes:

- not just the head
- adds an ordinal / contrastive regularizer
- in the full paper line, it sits with richer hierarchical models and richer
  feature spaces

Why it can matter:

- pronunciation labels are ordinal, not plain regression targets
- phoneme representations may benefit from explicit phoneme-aware geometry

What we learned:

- on our narrow 42-d `P001` feature stack, the loss alone does little
- that suggests the big gains come from the richer contract, not the loss by
  itself

Primary source:

- ConPCO abstract: <https://arxiv.org/abs/2406.02859>

### HierTFR / HierCB

What they change:

- hierarchical modeling across phoneme, word, and utterance
- correlation-aware / structure-aware scoring
- often paired with richer features and pretraining

Why they can matter:

- human pronunciation scoring is naturally hierarchical
- phone errors roll up into word-level and utterance-level judgments

Primary source:

- ACL Anthology abstract: <https://aclanthology.org/2024.acl-long.95/>

### HMamba

What it changes in the paper:

- replaces transformer blocks with Mamba selective state-space blocks
- jointly addresses APA and MDD
- uses a hierarchical setup and a specialized loss (`deXent`)

Why it can matter:

- longer-context sequence modeling with different inductive bias than a
  transformer
- joint APA + MDD may regularize the latent representation

What we learned locally:

- the **phone-level HMamba adaptation** inside `P001` is basically parity with
  `original + GOPT`, not a clear upgrade
- that makes the simple scorer swap uninteresting by itself
- the faithful hierarchical HMamba version still belongs in `P002`

Primary source:

- HMamba abstract: <https://arxiv.org/abs/2502.07575>

### HiPAMA

What it changes:

- explicitly models hierarchy across phoneme, word, and utterance
- uses multi-aspect attention so pronunciation aspects at the same linguistic
  level can inform each other

Why it can matter:

- "accuracy", "fluency", and "completeness" are not independent
- word- and utterance-level judgments should not be modeled as flat outputs

Why it belongs in `P002`:

- this is not a simple drop-in phone-level head
- it wants richer labels and richer structure

Primary source:

- HiPAMA abstract: <https://arxiv.org/abs/2211.08102>

## 4. Is changing the scorer just fixing the test?

Sometimes it can drift that way. The defense against that is experimental
discipline:

- hold the backbone fixed
- hold the dataset fixed
- hold the metric fixed
- change only one contract level at a time

That is why `P001` was useful:

- it established that a better downstream scorer really does matter under a
  fixed feature contract

And that is why `P002` should be explicit:

- once the feature/data contract changes, you are no longer asking the same
  question

So the right concern is not "never change the scorer." The right concern is
"do not mix scorer changes with richer inputs and then pretend the gain came
from one thing."

## 5. Where TIMIT fits

TIMIT is useful for:

- phone recognition prototyping
- alignment experiments
- duration-model experiments
- sanity checks on phonetic modeling

TIMIT is **not** a drop-in substitute for SpeechOcean762 in pronunciation
scoring because:

- it is mostly native read speech
- it has hand-labeled phones, but not the same non-native pronunciation scoring
  labels we care about

So TIMIT helps the acoustic/alignment side more than the final scorer-paper
question.


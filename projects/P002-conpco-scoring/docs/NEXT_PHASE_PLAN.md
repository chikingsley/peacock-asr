# P002 Next-Phase Plan

## 1. Positioning

`P002` is the direct continuation of `P001` in the same general research
genre.

`P001` asks:

- with fixed phoneme posteriors, what scorer family works best?

`P002` asks:

- once `P001` establishes the baseline scorer stack, which richer scoring
  ingredients actually add value?

That means `P002` is still a scoring-layer project. It is not a backbone paper
and it is not primarily a replication vanity project.

## 2. Current Evidence

The repo already has a strong Phase 1 conclusion:

- loss-only ConPCO on our 42-d GOP-SF feature stack gives negligible gain
  (`+0.003` PCC on GOPT/GOP-SF)
- ConPCO shows a bigger effect on HierCB-style richer-feature settings
  (`+0.0137` in the v4 reproduction)

Interpretation:

- the ConPCO loss is not the main story on our narrow feature stack
- the richer feature space and architecture are doing most of the work

So the next question is not "can we rerun loss-only ablations forever?"
The next question is:

- what is the minimum richer-feature extension that actually beats the `P001`
  baseline cleanly?

## 3. Recommended Immediate Goal

Treat `P002` as the richer-feature scoring paper:

- keep the `P001` scorer comparison as the base contract
- add features incrementally
- keep architecture changes secondary until feature gains are measured

Treat `P002` as the richer-contract scorer lane too:

- faithful HMamba does not belong in `P001`
- HiPAMA does not belong in `P001`
- both should be compared only after the required richer input/data contract is
  made explicit here

That yields a defensible paper question:

- Do richer features and ConPCO-style objectives materially improve
  pronunciation scoring beyond the `P001` baseline?

## 4. Next Experiment Ladder

### Phase 2A: Duration

Add one cheap timing feature family first.

Question:

- does explicit duration signal improve over GOP-SF-only features?

Why first:

- cheap to add
- interpretable
- low compute risk

### Phase 2B: Energy

Add energy statistics on top of duration.

Question:

- do simple prosodic / amplitude statistics add signal beyond duration?

Why second:

- still cheap
- helps separate "richer features help" from "only SSL helps"

### Phase 2C: Single SSL model

Add one SSL embedding family, not all three at once.

Question:

- how much gain comes from one strong SSL feature source alone?

Why:

- establishes whether the big feature jump is real before committing to the
  full HierCB feature burden

### Phase 3: Architecture only if justified

Only move to HierCB-style or branch-style architecture changes if the feature
experiments show a meaningful gain.

Reason:

- otherwise the paper stops being interpretable
- architecture and features become confounded immediately

### Phase 3B: Faithful advanced scorers

Once the richer feature/data contract is explicit, evaluate faithful scorer
families that are not simple `P001` drop-ins:

- HMamba (faithful hierarchical version)
- HiPAMA

These should share the same enriched feature set and supervision contract so
the comparisons are meaningful.

## 5. Comparison Contract

Every `P002` run should be compared against:

- `P001` best feature-based baseline
- `P001` best GOPT mean
- the ConPCO reproduction checkpoints already in this workspace

This keeps the paper honest. The goal is not to compare only against older
published numbers. The goal is to compare against our own strongest current
baseline under the same repo contract.

## 6. Deliverable Shape

The clean `P002` paper should probably land as:

- Table 1: `P001` inherited baseline
- Table 2: loss-only ablation summary
- Table 3: feature enrichment ladder
- Table 4: optional architecture follow-up, only if Phase 2 justifies it

## 7. Practical Takeaway

The repo should act as if `P002` is no longer asking whether ConPCO as a loss
is magically sufficient.

That question has already been answered well enough:

- on our narrow feature stack, no

The active question is:

- what richer scoring stack is actually worth the added complexity?

Current non-priority:

- the exploratory phone-level HMamba scorer swap in `P001`
  (`PCC 0.6341` on `original`, seed `501`) is useful as a local check, but it
  is not the main research direction

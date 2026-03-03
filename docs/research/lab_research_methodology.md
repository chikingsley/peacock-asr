# Lab Research Methodology: Rigorous Evaluation of ML/AI Research

*A living reference for how we read, evaluate, and conduct research.*

---

## 1. Ablation Studies: Isolating What Actually Works

Ablation studies systematically remove or disable individual components of a model to measure each one's contribution to overall performance. The term originates from neuroscience, where scientists removed parts of animal brains to study the effect on behavior. In ML, we do the same thing to neural network components.

### Why This Matters

A paper might introduce five changes simultaneously — a new normalization layer, a different activation function, a modified attention mechanism, a new learning rate schedule, and a larger training dataset — and report a 3% accuracy gain. Without ablations, you have no idea whether one change drove 100% of the gain or whether some changes actually *hurt* performance but were masked by others.

### What Good Ablation Tables Look Like

A proper ablation table starts from a baseline and adds (or removes) one change at a time, reporting performance at each step. The ConvNeXt paper (Liu et al., 2022, "A ConvNet for the 2020s") is one of the best examples of this in recent literature. The authors started from a standard ResNet-50, then made a series of incremental changes — macro design, ResNeXt-ification, inverted bottleneck, large kernel size, and micro design choices — reporting accuracy at each step while controlling FLOPs. This made it clear exactly how much each modernization contributed.

### What to Look For

When reading a paper, check:

- Does it present a full ablation table, or only the final combined result?
- Are components removed one at a time from the full system (top-down ablation) *and/or* added one at a time to a baseline (bottom-up)?
- Are the ablation experiments run with the same hyperparameters, random seeds, and data splits as the main result?
- Are interactions between components explored? (Sometimes component A only helps in the presence of component B.)

### What to Watch Out For

- **Cherry-picked ablations.** Authors may only ablate the components that look good when removed and skip the ones that don't clearly help.
- **Missing error bars.** A 0.2% improvement means nothing if the standard deviation across seeds is 0.5%.
- **Ablations at a different scale.** Sometimes ablations are run on a smaller model or dataset than the main results, which can change conclusions.

### Key Reference

Meyes et al. (2019), "Ablation Studies in Artificial Neural Networks" (arXiv:1901.08644) provides a formal treatment of ablation methodology, drawing the analogy to neuroscience and demonstrating approaches on benchmark vision tasks.

---

## 2. Compute-Fair Comparisons: Controlling for Resources

Performance gains in ML often come from simply using more compute — more GPUs, longer training, bigger batches — rather than from architectural innovation. Any claim of improvement must be evaluated against a compute-equivalent baseline.

### The Core Problem

If Model A trains for 300 epochs on 64 GPUs and Model B (the "old" baseline) trained for 90 epochs on 8 GPUs, comparing their accuracy is meaningless as an architecture comparison. You're measuring a resource difference, not a design difference.

### Landmark Examples

**ConvNets Match Vision Transformers at Scale** (Smith et al., 2023, arXiv:2310.16764). This paper from Google DeepMind demonstrated that NFNet (a pure CNN) matches Vision Transformer performance when both are given comparable compute budgets. After fine-tuning on ImageNet, NFNets achieved 90.4% top-1 accuracy, matching ViTs at similar FLOPs. This was a direct challenge to the narrative that transformers are architecturally superior for vision — it turned out much of the gap was a compute gap.

**ConvNeXt** (Liu et al., 2022). The authors explicitly controlled FLOPs throughout their modernization process, ensuring each design change was evaluated at roughly equivalent computational cost. This is the gold standard for how to present architectural comparisons.

**Chinchilla Scaling Laws** (Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models," arXiv:2203.15556). DeepMind trained over 400 language models and showed that for a given compute budget, model size and training tokens should be scaled equally. Their 70B parameter Chinchilla model outperformed the 280B parameter Gopher using the *same* compute budget — simply by allocating resources differently. This paper fundamentally changed how the field thinks about the relationship between compute, data, and model size.

### What to Check in Papers

- **Are FLOPs or GPU-hours reported** for both the proposed method and all baselines?
- **Were baselines retrained** under the same conditions (same training duration, same hardware, same optimizer tuning), or are the authors comparing against stale numbers from the original paper?
- **Is the training data identical** across all comparisons?
- **Is the hyperparameter search budget equivalent?** If the new model got 200 HPO trials and the baseline got 10, you're comparing tuning effort, not architecture.

### Scaling Laws as a Lens

After Chinchilla, we know that for transformers, performance is roughly a function of total compute (≈ 6 × parameters × tokens). Use this as a quick sanity check: if a paper claims a smaller model beats a larger one, check whether the smaller model saw proportionally more data. If so, the comparison may be compute-equivalent, and the result is about efficiency. If the smaller model saw *less* data *and* beats the larger model, that's more interesting.

---

## 3. Reproducibility: Can Anyone Actually Replicate This?

The ML reproducibility crisis is well-documented. A study by Gundersen and Kjensmo found that none of 400 papers from leading conferences satisfied all reproducibility criteria, with most satisfying only 20–30%. Raff (2019) attempted to reproduce results from 255 papers using only the paper's text and found that 93 could not be reproduced.

### The NeurIPS/ICML Reproducibility Checklist

Since 2019, NeurIPS has required a reproducibility checklist with submissions. After its introduction, papers including reproducibility materials rose from roughly 50% to over 75%. The checklist asks authors to confirm they've provided:

- Code, data, and instructions to reproduce main results
- All training details: data splits, hyperparameters, how they were chosen
- Error bars from multiple runs with different random seeds
- Compute resource details: GPU type, memory, wall-clock time
- Total compute for the research project (including failed experiments)

### What We Should Check

When evaluating a paper for our work:

- **Is code released?** And does the README actually explain how to reproduce the main table?
- **Are random seeds reported?** Neural network training is stochastic. Results from a single seed are anecdotal.
- **Are error bars or confidence intervals shown?** If not, we don't know whether differences are statistically significant.
- **Is the full compute budget disclosed?** Including preliminary experiments that didn't make it into the paper. This tells us the true cost of the research.

### What We Should Do in Our Own Work

- Always run experiments with at least 3 different random seeds and report mean ± standard deviation.
- Pin all dependency versions (Python, PyTorch, CUDA, etc.) in a requirements file.
- Log all hyperparameters using experiment tracking tools (Weights & Biases, MLflow, etc.).
- Document the total compute used, including failed runs.
- Make the random number generator state explicit in code.

---

## 4. Baseline and Benchmark Integrity

### Common Failure Modes

- **Stale baselines.** Comparing against numbers reported in a 3-year-old paper, when those methods have since been improved with better training recipes, data augmentation, and hyperparameter tuning.
- **Inconsistent evaluation protocols.** Different data splits, preprocessing pipelines, or evaluation metrics across methods.
- **Benchmark overfitting.** The community has collectively overfit to popular benchmarks like ImageNet. A model tuned for ImageNet may not generalize to other domains — the ConvNeXt backbone comparison study found that models excelling on natural images performed poorly in the medical domain.
- **Leakage between train and test.** Particularly in time-series and medical data, where cross-validation without temporal splitting leads to inflated performance.

### What We Should Do

- When comparing methods, retrain baselines ourselves under identical conditions whenever feasible.
- Use multiple benchmarks spanning different domains to assess generalization.
- Be skeptical of SOTA claims on a single benchmark.
- Check whether the test set was truly held out or whether any decisions (architecture search, hyperparameter tuning) used test set performance.

---

## 5. The Paper Review Checklist

Use this when reading *any* ML paper. Not every paper needs to pass every check, but failures should be noted and factored into how much weight we give the results.

**Ablation & Attribution**
- [ ] Is there a full ablation table isolating each proposed change?
- [ ] Are ablations run at the same scale as the main experiments?
- [ ] Are error bars reported for ablation results?
- [ ] Do the individual ablation gains sum roughly to the total reported gain? (If not, there are interaction effects worth understanding.)

**Compute Fairness**
- [ ] Are FLOPs, GPU-hours, or wall-clock time reported for all methods?
- [ ] Were baselines given equivalent compute budgets (training time, HPO budget)?
- [ ] Is the training data identical across all methods being compared?
- [ ] Would the Chinchilla scaling lens explain the gains (more data or compute rather than better architecture)?

**Reproducibility**
- [ ] Is code released with a working README?
- [ ] Are all hyperparameters, data splits, and preprocessing steps documented?
- [ ] Are results reported over multiple random seeds with error bars?
- [ ] Is the total compute budget (including failed experiments) disclosed?
- [ ] Are dependency versions pinned?

**Benchmark & Baseline Integrity**
- [ ] Are baselines current (retrained or from recent papers)?
- [ ] Is the evaluation protocol consistent across all methods?
- [ ] Are results shown on multiple benchmarks/datasets?
- [ ] Is there evidence of generalization beyond the primary benchmark?

---

## 6. Our Lab Standards for Publishing

When we publish our own work, we hold ourselves to these standards:

1. **Always include ablation tables.** Every component we introduce must be justified with an ablation. Follow the ConvNeXt model: incremental additions with controlled compute at each step.

2. **Report compute budgets.** Total FLOPs, GPU-hours, hardware specs, and wall-clock time for every experiment, including the cost of hyperparameter search.

3. **Multiple seeds, always.** Minimum 3 seeds for all reported results. Report mean ± std. If compute is too expensive for multiple seeds on the full model, run multi-seed experiments at a smaller scale and single-seed at full scale, but be transparent about it.

4. **Retrain baselines.** Never compare against numbers pulled from another paper without verifying them under our conditions. If we can't retrain, explicitly note this limitation.

5. **Release code and configs.** Every paper we publish should have a companion repository with: training code, evaluation code, configuration files, and a README that reproduces the main results table.

6. **Use the NeurIPS reproducibility checklist** as a pre-submission self-audit, regardless of which venue we're targeting.

---

## Key References

| Paper | Year | Why It Matters |
|-------|------|---------------|
| Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla) | 2022 | Established that model size and data should scale equally for a given compute budget |
| Liu et al., "A ConvNet for the 2020s" (ConvNeXt) | 2022 | Gold standard for incremental ablation with controlled FLOPs |
| Smith et al., "ConvNets Match Vision Transformers at Scale" | 2023 | Showed CNNs match ViTs when given equal compute |
| Meyes et al., "Ablation Studies in Artificial Neural Networks" | 2019 | Formal treatment of ablation methodology |
| Pineau et al., "Improving Reproducibility in ML Research" | 2021 | NeurIPS reproducibility program and checklist |
| REFORMS Checklist (Kapoor et al.) | 2024 | Consensus recommendations for ML-based science |
| Sardana et al., "Beyond Chinchilla-Optimal" | 2024 | Extends scaling laws to account for inference costs |

---

*Last updated: March 2026. This is a living document — update as we encounter new methodological insights, papers, or failure modes.*

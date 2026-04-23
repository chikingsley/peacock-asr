---
arxiv: "2306.02061"
title: "Balancing Logit Variation for Long-tailed Semantic Segmentation"
authors:
  - "Yuchao Wang"
  - "Jingjing Fei"
  - "Haochen Wang"
  - "Wei Li"
  - "Tianpeng Bao"
  - "Liwei Wu"
  - "Rui Zhao"
  - "Yujun Shen"
citation_author: "Wang et al."
year: 2023
doi: null
venue: "CVPR 2023"
source_pdf: "paper.pdf"
extraction_method: "Manual extraction from CVPR PDF and arXiv LaTeX source (2306.02061v1)"
extracted_at: "2026-03-23"
llm_friendly: true
tags: [class-imbalance, long-tail, logit-perturbation, noise-injection, semantic-segmentation]
---

## Metadata

- Authors: Yuchao Wang, Jingjing Fei, Haochen Wang, Wei Li, Tianpeng Bao, Liwei Wu, Rui Zhao, Yujun Shen
- Affiliations: Shanghai Jiao Tong University, SenseTime Research, Chinese Academy of Sciences, CUHK
- arXiv: 2306.02061
- Code: <https://github.com/grantword8/BLV>
- Year: 2023, CVPR

## TL;DR

Add category-frequency-scaled Gaussian noise to logits during training to expand the feature-space footprint of rare (tail) classes. The noise magnitude is inversely proportional to class frequency, so tail classes get more perturbation. This is a training-only regularizer: the noise is removed at inference. Despite being trivially simple to implement (~5 lines of PyTorch), BLV consistently improves mIoU on tail classes across fully-supervised, semi-supervised, and UDA segmentation settings.

## Abstract

Semantic segmentation usually suffers from a long-tail data distribution. Due to the imbalanced number of samples across categories, the features of those tail classes may get squeezed into a narrow area in the feature space. Towards a balanced feature distribution, the authors introduce category-wise variation into the network predictions in the training phase such that an instance is no longer projected to a feature point, but a small region instead. Such a perturbation is highly dependent on the category scale, which appears as assigning smaller variation to head classes and larger variation to tail classes. The introduced variation is discarded at the inference stage to facilitate a confident prediction.

## Core Method: Balanced Logit Variation (BLV)

### Problem Setup

Given a model that outputs logits `z_k^i` for instance (pixel) `i` and category `k`, the standard cross-entropy loss is computed from softmax probabilities:

```text
p_k^i = exp(z_k^i) / sum_j exp(z_j^i)
L_CE = - sum_k y_k^i * log(p_k^i)
```

Because the logit dimensionality equals the number of categories and directly determines the categorical area in feature space, manipulating logits is the most direct way to address the long-tail squeeze.

### The BLV Formula (Equation 3)

The perturbed logit is:

```text
z_hat_k^i = z_k^i + (c_k / max_i(c_i)) * |delta(sigma)|
```

where:

```text
c_k = log( sum_j(q_j) / q_k )
```

- `q_k` = number of instances (pixels) with category `k`
- `delta(sigma)` = sample from N(0, sigma) -- Gaussian with mean 0, standard deviation sigma
- `|delta(sigma)|` = absolute value of the sample (so the noise is always non-negative)
- `c_k / max(c_i)` = category balance coefficient, normalized to [0, 1]

**Key properties:**

- Head classes (large `q_k`) get small `c_k`, so small perturbation
- Tail classes (small `q_k`) get large `c_k`, so large perturbation
- The absolute value ensures noise is always additive (positive direction)
- The `max` normalization ensures the coefficient stays in [0, 1]

### Clamping (from Supplementary Material)

The `|delta(sigma)|` term is clamped to [0, 1] to avoid particularly large values that make training unstable:

```python
noise = sampler.sample(pred.shape).clamp(0, 1)
```

**IMPORTANT implementation detail:** In the actual pseudo-code (Algorithm 1 in Supplementary), the clamping is applied to the raw sample with `.clamp(0, 1)` (not `.abs().clamp(-1, 1)`). The `.clamp(0, 1)` on a N(0, sigma) sample effectively takes only the positive part and caps it at 1. Then `.abs()` is applied afterward, which is redundant for values already in [0, 1] but is present in the code. The net effect is: sample from N(0, sigma), clamp to [0, 1], take absolute value.

### Hyperparameter: sigma

The only hyperparameter is sigma, the standard deviation of the Gaussian noise.

**Default values used in the paper:**

- sigma = 6 for all tasks EXCEPT:
- sigma = 4 for UDA segmentation under the SYNTHIA -> Cityscapes setting

**Ablation over sigma (Table 7):**

| Baseline | sigma=3 | sigma=4 | sigma=5 | sigma=6 | sigma=7 |
|----------|---------|---------|---------|---------|---------|
| **GTA5 -> Cityscapes** | | | | | |
| 55.9 | 58.0 | 58.8 | 58.2 | **59.0** | 58.7 |
| **SYNTHIA -> Cityscapes** | | | | | |
| 52.7 | 56.5 | **56.8** | 55.9 | 56.3 | 56.1 |

The method is robust to sigma within {3..7}; all values beat the baseline.

### Pseudo-code (from Supplementary, Algorithm 1)

```python
# frequency_list: a list containing the frequency of pixels of each category.
#   This is c_k / max(c_k), i.e., the normalized log-inverse-frequency.
# pred: model output logits, shape [B, C, H, W]
# target: ground-truth label
# sigma: hyper-parameter

def BLV_Loss(pred, target, sigma, frequency_list):

    sampler = torch.distributions.normal.Normal(0, sigma)

    noise = sampler.sample(pred.shape).clamp(0, 1).to(pred.device)

    pred = pred + (noise.abs().permute(0, 2, 3, 1)
                   * frequency_list
                   / frequency_list.max()
                  ).permute(0, 3, 1, 2)

    loss = torch.nn.functional.cross_entropy(pred, target)

    return loss
```

**Note on `frequency_list`:** Despite the variable name, the pseudo-code shows it is multiplied by `frequency_list / frequency_list.max()`, which corresponds to `c_k / max(c_i)` from Eq. 3. So `frequency_list` here contains the `c_k = log(sum(q_j) / q_k)` values, NOT the raw frequencies.

## Ablation: Form of Variation (Table 6)

Different noise distributions were tested (DAFormer on GTA5 -> Cityscapes):

| Variation | mIoU |
|-----------|------|
| None (baseline) | 55.9 |
| **Gaussian** | **59.0** (+3.1) |
| Uniform [0, 1] | 58.2 (+2.3) |
| Beta (alpha=0.5, beta=0.5) | 57.9 (+2.0) |
| Exponential (lambda=1) | 58.5 (+2.6) |

All perturbation forms are clipped to [0, 1]. Gaussian is best, but all forms improve over the baseline, suggesting the key ingredient is the frequency-dependent scaling, not the specific noise distribution.

## Ablation: Components (Table 8)

| Setting | Baseline | w/o variation | w/o balance | BLV |
|---------|----------|--------------|-------------|-----|
| GTA5 -> Cityscapes | 55.9 | 56.5 | 56.8 | **59.0** |
| SYNTHIA -> Cityscapes | 52.7 | 53.9 | 54.5 | **56.8** |

- "w/o variation" = no noise, just add a constant category-frequency adjustment
- "w/o balance" = add noise uniformly, without category-frequency scaling

Both components help individually, but their combination (full BLV) is substantially better.

## Results Summary

### Fully Supervised Segmentation (Cityscapes val)

| Backbone | Decoder | mIoU | mIoU (tail) |
|----------|---------|------|-------------|
| HRNet-18 | OCRHead | 79.22 | 63.51 |
| + BLV | | 79.94 (+0.72) | 66.70 (+3.19) |
| ResNet50 | UperHead | 78.28 | 62.56 |
| + BLV | | 78.63 (+0.35) | 64.57 (+2.01) |
| ResNet50 | PSPHead | 77.98 | 61.96 |
| + BLV | | 78.53 (+0.55) | 63.34 (+1.38) |
| ResNet101 | UperHead | 79.41 | 64.68 |
| + BLV | | 79.88 (+0.47) | 66.29 (+1.61) |
| MiT-b0 | SegformerHead | 76.85 | 67.58 |
| + BLV | | 77.09 (+0.24) | 68.91 (+1.33) |
| Swin-T | K-NeT | 79.68 | 71.70 |
| + BLV | | 80.11 (+0.43) | 72.94 (+1.24) |
| Vit-B16 | UperHead | 76.48 | 68.25 |
| + BLV | | 77.68 (+1.20) | 70.63 (+2.38) |

Tail categories: Wall, T.light, Sign, Rider, Truck, Bus, Train, M.bike, Bike (9 categories).

### Semi-Supervised Segmentation (Cityscapes val, Self-Training baseline)

| Partition | mIoU | mIoU (tail) |
|-----------|------|-------------|
| 1/16 (186 labeled) | 68.21 -> 69.26 (+1.05) | 53.09 -> 55.23 (+2.14) |
| 1/8 (372) | 72.01 -> 73.27 (+1.26) | 58.74 -> 60.33 (+1.59) |
| 1/4 (744) | 74.03 -> 75.52 (+1.49) | 61.76 -> 63.51 (+1.75) |
| 1/2 (1488) | 77.99 -> 78.98 (+0.99) | 65.96 -> 67.24 (+1.28) |

### UDA Segmentation (GTA5 -> Cityscapes)

| Method | mIoU |
|--------|------|
| DAFormer (CNN) | 55.9 -> 59.0 (+3.1) |
| DAFormer (Transformer) | 68.3 -> 69.6 (+1.3) |
| HRDA (Transformer) | 73.8 -> 74.9 (+1.1) |

### Comparison with Long-Tail Methods (Table 5)

| Setting | Method | mIoU | mIoU (tail) |
|---------|--------|------|-------------|
| Fully sup. | Logit-Adjustment | 75.9 | 62.4 |
| | Lovasz-Softmax | 76.6 | 63.9 |
| | **BLV** | **77.7** | **66.2** |
| Semi sup. | DARS | 72.8 | 58.4 |
| | **BLV** | **73.2** | **59.3** |
| UDA | CLAN | 45.9 | 28.5 |
| | CBST | 43.2 | 25.9 |
| | Logit-Adjustment | 56.5 | 41.9 |
| | **BLV** | **59.0** | **45.7** |

## Computational Overhead

| Backbone | Decoder | w/o BLV | w/ BLV |
|----------|---------|---------|--------|
| HRNet-18 | OCRHead | 20h11m | 21h07m (+4.6%) |
| ResNet50 | UperHead | 16h20m | 16h47m (+2.8%) |

Measured on 8x V100 GPUs. Negligible overhead.

## Key Implementation Details for P010 Adaptation

1. **The noise is applied to logits, not to features or probabilities.** It goes directly on the model's output before cross-entropy.

2. **Category frequency computation:** `c_k = log(sum(q_j) / q_k)` where `q_k` is the count of instances per class. For segmentation this is pixel counts. For pronunciation assessment, this would be phoneme occurrence counts.

3. **Normalization:** The `c_k` values are divided by `max(c_k)` so the balance coefficient is in [0, 1]. The rarest class gets coefficient 1.0.

4. **Noise generation:** Sample from N(0, sigma), clamp to [0, 1], multiply by the per-class balance coefficient. The result is added to the logit for that class.

5. **Training only:** The noise is NOT applied at inference time.

6. **The noise is the same across all classes for a given instance.** Looking at the pseudo-code, a single noise tensor is sampled for the full pred shape [B, C, H, W], then scaled per-class by the frequency coefficient. So each (batch, height, width) position gets its own noise draw, but the frequency scaling is applied per-channel (per-class).

7. **Semi-supervised extension:** When labels are not available for all data, estimate class frequencies from pseudo-labels and update every epoch.

8. **Temporal sigma scheduling (Supplementary):** A further improvement uses time-varying sigma that ramps up to sigma_0 at t_mid, then decreases to 0 at t_end. This gave +0.4 mIoU improvement over constant sigma (69.6 -> 70.0 on DAFormer GTA5->Cityscapes).

## Discussion and Limitations

- BLV requires knowing class frequencies. This is straightforward for fully-supervised settings but requires estimation (via pseudo-labels) for semi-supervised and UDA settings.
- The method is orthogonal to other long-tail techniques (oversampling, loss reweighting) and can be combined with them.
- All noise distributions tested (Gaussian, Uniform, Beta, Exponential) improve over baseline, but Gaussian works best.
- The method is model-agnostic: works with CNNs (ResNet) and Transformers (ViT, Swin-T, MiT).

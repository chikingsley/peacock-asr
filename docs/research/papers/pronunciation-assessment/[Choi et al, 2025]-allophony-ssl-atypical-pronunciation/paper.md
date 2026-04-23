---
arxiv: 2502.07029
title: "Leveraging Allophony in Self-Supervised Speech Models for Atypical Pronunciation Assessment"
authors: "Kwanghee Choi, Eunjung Yeo, Kalvin Chang, Shinji Watanabe, David Mortensen"
year: 2025
venue: "arxiv"
category: pronunciation-assessment
tags: [ssl, pronunciation-assessment, allophony, gop, gmm, atypical-speech, mixgop, phoneme-scoring, sota]
---

Introduces MixGoP, a pronunciation scoring method that fits Gaussian Mixture Models over frozen SSL features to capture allophonic variation (context-dependent phoneme realizations) rather than treating each phoneme as a single distribution. The key insight is that SSL representations already encode allophonic context, so modeling the feature space with GMMs rather than a single Gaussian per phoneme better captures canonical pronunciation boundaries. Claims SOTA on 4 out of 5 pronunciation assessment datasets, making it highly relevant to the current GOP-based scoring pipeline and a strong baseline to compare against.

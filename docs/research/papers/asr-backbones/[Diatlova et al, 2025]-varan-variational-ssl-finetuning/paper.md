---
arxiv: 2508.12061
title: "VARAN: Variational Inference for Self-Supervised Speech Models Fine-Tuning on Downstream Tasks"
authors: "Daria Diatlova, Ivan Medennikov, Maxim Koreeda, Aleksei Romanenko"
year: 2025
venue: "arxiv"
category: asr-backbones
tags: [ssl, layer-aggregation, variational-inference, layer-weighting, fine-tuning, dynamic-weighting]
---

Proposes VARAN, a framework that uses variational inference to learn dynamic, input-dependent layer weights when fine-tuning SSL speech models on downstream tasks, as opposed to static learned scalar weights (as in weighted-sum approaches). By treating layer weights as latent variables inferred from the input, the model can adaptively emphasize different layers depending on the specific utterance's characteristics. This is directly relevant to pronunciation assessment as input-dependent layer weighting could allow a scoring model to selectively use phonetically-discriminative layers based on the specific phoneme or speaker context being evaluated.

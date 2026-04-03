from __future__ import annotations

import torch
import torch.nn.functional as F

from p012.interfaces import InterfaceFeaturizer, infer_hconv_output_dim


class DummyUpstream:
    def __init__(self, num_layers: int = 4, hidden_size: int = 16, stride: int = 160):
        self.num_layers = num_layers
        self.hidden_sizes = [hidden_size] * num_layers
        self.downsample_rates = [stride] * num_layers
        self.stride = stride
        self.hidden_size = hidden_size

    def __call__(self, wavs: torch.FloatTensor, wavs_len: torch.LongTensor):
        if wavs.dim() == 3:
            wavs = wavs.squeeze(-1)

        max_len = int(torch.div(wavs_len.max() - 1, self.stride, rounding_mode="floor") + 1)
        total_samples = max_len * self.stride
        padded = F.pad(wavs, (0, total_samples - wavs.shape[1]))
        frames = padded.view(wavs.shape[0], max_len, self.stride).mean(dim=-1, keepdim=True)
        h_len = torch.div(wavs_len - 1, self.stride, rounding_mode="floor") + 1
        all_hs = [frames.repeat(1, 1, self.hidden_size) + layer_id for layer_id in range(self.num_layers)]
        all_lens = [h_len.clone() for _ in range(self.num_layers)]
        return all_hs, all_lens


def test_infer_hconv_output_dim_for_hubert_base_shape():
    assert infer_hconv_output_dim(upstream_layer_num=13, upstream_feat_dim=768) == 768


def test_weighted_sum_featurizer_matches_upstream_hidden_size():
    upstream = DummyUpstream()
    featurizer = InterfaceFeaturizer(upstream, interface="weighted_sum")

    wavs = torch.randn(2, 16000)
    wavs_len = torch.tensor([16000, 12000])
    all_hs, all_lens = upstream(wavs, wavs_len)

    hs, hs_len = featurizer(all_hs, all_lens)

    assert hs.shape[0] == 2
    assert hs.shape[-1] == featurizer.output_size
    assert torch.equal(hs_len, all_lens[0])


def test_hconv_featurizer_uses_official_output_width():
    upstream = DummyUpstream()
    featurizer = InterfaceFeaturizer(upstream, interface="hconv")

    wavs = torch.randn(2, 16000)
    wavs_len = torch.tensor([16000, 12000])
    all_hs, all_lens = upstream(wavs, wavs_len)

    hs, hs_len = featurizer(all_hs, all_lens)

    assert hs.shape[0] == 2
    assert hs.shape[-1] == featurizer.output_size
    assert hs.shape[-1] == infer_hconv_output_dim(upstream.num_layers, upstream.hidden_sizes[0])
    assert torch.equal(hs_len, all_lens[0])

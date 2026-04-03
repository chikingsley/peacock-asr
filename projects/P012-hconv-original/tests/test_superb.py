from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F

from p012.superb import InterfaceSuperbER, InterfaceSuperbPR
from p012.vendor import ensure_third_party_on_path

ensure_third_party_on_path()


class DummyUpstream:
    def __init__(self, num_layers: int = 4, hidden_size: int = 16, stride: int = 160):
        self.num_layers = num_layers
        self.hidden_sizes = [hidden_size] * num_layers
        self.downsample_rates = [stride] * num_layers
        self.stride = stride
        self.hidden_size = hidden_size

    def to(self, device):
        return self

    def eval(self):
        return self

    def train(self):
        return self

    def parameters(self):
        return iter(())

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


def write_dummy_wavs(root: Path, seconds: list[int], sample_rate: int = 16000) -> list[str]:
    wav_paths: list[str] = []
    for index, second in enumerate(seconds):
        num_samples = second * sample_rate
        timeline = torch.linspace(0, second, num_samples, dtype=torch.float32)
        waveform = 0.1 * torch.sin(2 * torch.pi * 220 * timeline)
        wav_path = root / f"sample-{index}.wav"
        sf.write(wav_path, waveform.numpy(), sample_rate)
        wav_paths.append(str(wav_path))
    return wav_paths


def test_interface_superb_pr_smoke():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        wav_paths = write_dummy_wavs(tempdir_path, [10, 2, 1, 8, 5])

        class TestPR(InterfaceSuperbPR):
            def default_config(self) -> dict:
                config = super().default_config()
                config["prepare_data"] = {}
                return config

            def build_upstream(self, build_upstream: dict):
                return DummyUpstream()

            def prepare_data(
                self,
                prepare_data: dict,
                target_dir: str,
                cache_dir: str,
                get_path_only: bool = False,
            ):
                df = pd.DataFrame(
                    data={
                        "id": list(range(len(wav_paths))),
                        "wav_path": wav_paths,
                        "transcription": [
                            "HH AH L OW",
                            "F AY N",
                            "OW",
                            "AY TH IH NG K IH Z G UH D",
                            "M EY B IY OW K EY",
                        ],
                    }
                )
                train_path = Path(target_dir) / "train.csv"
                valid_path = Path(target_dir) / "valid.csv"
                test_path = Path(target_dir) / "test.csv"
                df.iloc[:3].to_csv(train_path, index=False)
                df.iloc[3:4].to_csv(valid_path, index=False)
                df.iloc[4:].to_csv(test_path, index=False)
                return train_path, valid_path, [test_path]

        problem = TestPR()
        config = problem.default_config()
        config["target_dir"] = tempdir
        config["device"] = "cpu"
        config["train"]["total_steps"] = 2
        config["train"]["log_step"] = 1
        config["train"]["eval_step"] = 1
        config["train"]["save_step"] = 1
        config["eval_batch"] = 1
        config["build_upstream"]["name"] = "dummy"
        config["build_featurizer"]["interface"] = "hconv"
        config["build_featurizer"]["output_dim"] = 16
        problem.run(**config)


def test_interface_superb_er_weighted_sum_smoke():
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir_path = Path(tempdir)
        wav_paths = write_dummy_wavs(tempdir_path, [10, 2, 1, 8, 5])

        class TestER(InterfaceSuperbER):
            def default_config(self) -> dict:
                config = super().default_config()
                config["prepare_data"] = {}
                return config

            def build_upstream(self, build_upstream: dict):
                return DummyUpstream()

            def prepare_data(
                self,
                prepare_data: dict,
                target_dir: str,
                cache_dir: str,
                get_path_only: bool = False,
            ):
                df = pd.DataFrame(
                    data={
                        "id": [Path(path).stem for path in wav_paths],
                        "wav_path": wav_paths,
                        "label": ["a", "b", "a", "c", "d"],
                    }
                )
                train_csv = Path(target_dir) / "train.csv"
                valid_csv = Path(target_dir) / "valid.csv"
                test_csv = Path(target_dir) / "test.csv"
                df.to_csv(train_csv, index=False)
                df.to_csv(valid_csv, index=False)
                df.to_csv(test_csv, index=False)
                return train_csv, valid_csv, [test_csv]

        problem = TestER()
        config = problem.default_config()
        config["target_dir"] = tempdir
        config["device"] = "cpu"
        config["train"]["total_steps"] = 2
        config["train"]["log_step"] = 1
        config["train"]["eval_step"] = 1
        config["train"]["save_step"] = 1
        config["eval_batch"] = 1
        config["build_upstream"]["name"] = "dummy"
        config["build_featurizer"]["interface"] = "weighted_sum"
        problem.run(**config)

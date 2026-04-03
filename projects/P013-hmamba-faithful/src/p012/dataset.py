from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class MispronunciationStats:
    num_correct: int
    num_mispronounced: int

    @property
    def mis_weight_ratio(self) -> float:
        if self.num_mispronounced <= 0:
            return 0.0
        return self.num_correct / self.num_mispronounced


class GoPDataset:
    def __init__(
        self,
        set_name: str,
        data_dir: str | Path | None = None,
        data_dir2: str | None = None,
        data_dir3: str | Path | None = None,
        am: str = "librispeech",
        mode: str = "apa",
    ) -> None:
        if am == "librispeech":
            norm_mean, norm_std = 3.203, 4.045
        elif am == "paiia":
            norm_mean, norm_std = -0.652, 9.737
        elif am == "paiib":
            norm_mean, norm_std = -0.516, 9.247
        else:
            raise ValueError("Acoustic Model Unrecognized.")

        self.mode = mode
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.feat2: list[torch.Tensor] = []
        self.feat3: torch.Tensor | None = None

        root = Path(data_dir) if data_dir is not None else None
        if root is None:
            raise ValueError("data_dir is required.")

        set_dir = root / set_name
        self.feat = torch.tensor(np.load(set_dir / "feat.npy"), dtype=torch.float32)
        self.feat = self._norm_valid(self.feat, norm_mean, norm_std)

        if data_dir2:
            self.feat2 = [
                torch.tensor(np.load(Path(data) / set_name / "feat.npy"), dtype=torch.float32)
                for data in data_dir2.split()
            ]
        if data_dir3:
            self.feat3 = torch.tensor(np.load(Path(data_dir3) / set_name / "feat.npy"), dtype=torch.float32)

        self.phn_label = torch.tensor(np.load(set_dir / "label_phn.npy"), dtype=torch.float32)
        if self.mode == "apa":
            self.utt_label = torch.tensor(np.load(set_dir / "label_utt.npy"), dtype=torch.float32)
            self.word_label = torch.tensor(np.load(set_dir / "label_word.npy"), dtype=torch.float32)
            self.utt_label = self.utt_label / 5
            self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5

        self.utt_id = np.load(set_dir / "utt_id.npy", allow_pickle=True)

    def _norm_valid(self, feat: torch.Tensor, norm_mean: float, norm_std: float) -> torch.Tensor:
        norm_feat = torch.zeros_like(feat)
        valid = feat[:, :, 0] != 0
        norm_feat[valid] = (feat[valid] - norm_mean) / norm_std
        return norm_feat

    def mispronunciation_stats(self) -> MispronunciationStats:
        canophn = self.phn_label[:, :, 0]
        realphn = self.phn_label[:, :, 1]
        valid = realphn >= 0
        correct = valid & (canophn == realphn)
        mispronounced = valid & (canophn != realphn)
        return MispronunciationStats(
            num_correct=int(correct.sum().item()),
            num_mispronounced=int(mispronounced.sum().item()),
        )

    def __len__(self) -> int:
        return int(self.feat.shape[0])

    def __getitem__(self, idx: int) -> tuple[object, ...]:
        feat2 = [x[idx, :] for x in self.feat2] if self.data_dir2 else []
        feat3 = self.feat3[idx, :] if self.feat3 is not None else []
        if self.mode == "apa":
            return (
                self.feat[idx, :],
                feat2,
                feat3,
                self.phn_label[idx, :, 0],
                self.phn_label[idx, :, 1],
                self.phn_label[idx, :, 2],
                self.phn_label[idx, :, -1],
                self.word_label[idx, :],
                self.utt_label[idx, :],
                self.utt_id[idx],
            )
        if self.mode == "mdd":
            return (
                self.feat[idx, :],
                feat2,
                feat3,
                self.phn_label[idx, :, 0],
                self.phn_label[idx, :, 1],
                self.phn_label[idx, :, 2],
                self.utt_id[idx],
            )
        raise ValueError(f"Dataset mode must be 'apa' or 'mdd', got {self.mode}.")

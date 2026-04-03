from __future__ import annotations

import torch


def require_cuda_device(context: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(f"{context} requires CUDA. Install a CUDA-enabled PyTorch stack and run on a CUDA device.")
    return torch.device("cuda")

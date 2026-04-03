from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .dataset import GoPDataset
from .models import HMamba
from .runtime import require_cuda_device
from .trainer import load_checkpoint, load_conf, load_phn_dict


def recog(audio_model: HMamba, val_loader: DataLoader, args: argparse.Namespace) -> None:
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = exp_dir / "models" / args.checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = exp_dir / "models" / "best_audio_model.pth"

    hyp_f = open(exp_dir / "hyp", "w", encoding="utf-8")
    rel_f = open(exp_dir / "rel", "w", encoding="utf-8")
    can_f = open(exp_dir / "can", "w", encoding="utf-8")
    hyp_nosil_f = rel_nosil_f = can_nosil_f = None
    if args.remove_sil:
        hyp_nosil_f = open(exp_dir / "hyp_nosil", "w", encoding="utf-8")
        rel_nosil_f = open(exp_dir / "rel_nosil", "w", encoding="utf-8")
        can_nosil_f = open(exp_dir / "can_nosil", "w", encoding="utf-8")

    id2phn = load_phn_dict(args.phn_dict)
    device = require_cuda_device("HMamba recognition")
    load_checkpoint(audio_model, checkpoint_path, device)
    audio_model = audio_model.to(device)
    audio_model.eval()

    with torch.no_grad():
        for audio_input, audio_input2, audio_input3, canophns, realphns, bies, utt_id in tqdm(val_loader):
            audio_input = audio_input.to(device)
            audio_input2 = [input2.to(device) for input2 in audio_input2]
            if isinstance(audio_input3, torch.Tensor):
                audio_input3 = audio_input3.to(device)
            canophns = canophns.to(device)
            realphns = realphns.to(device)
            bies = bies.to(device)
            utt_id, = utt_id

            pad_mask = realphns >= 0
            logits = audio_model(audio_input, audio_input2, audio_input3, canophns, bies, mask=pad_mask)[-1]
            hyp = torch.argmax(logits, dim=-1).cpu().detach()
            rel = realphns.cpu().detach()
            can = canophns.cpu().detach()
            mask_cpu = pad_mask.cpu()

            hyp_str = " ".join(id2phn[int(id_)] for id_ in hyp.masked_select(mask_cpu).tolist()).lower()
            rel_str = " ".join(id2phn[int(id_)] for id_ in rel.masked_select(mask_cpu).int().tolist()).lower()
            can_str = " ".join(id2phn[int(id_)] for id_ in can.masked_select(mask_cpu).int().tolist()).lower()

            if args.remove_special_token:
                hyp_str = " ".join(re.sub(args.special_token, "", hyp_str).split())
                rel_str = " ".join(re.sub(args.special_token, "", rel_str).split())
                can_str = " ".join(re.sub(args.special_token, "", can_str).split())

            hyp_f.write(f"{utt_id} {hyp_str}\n")
            rel_f.write(f"{utt_id} {rel_str}\n")
            can_f.write(f"{utt_id} {can_str}\n")

            if args.remove_sil and hyp_nosil_f and rel_nosil_f and can_nosil_f:
                hyp_nosil_f.write(f"{utt_id} {' '.join(re.sub(r'sil', '', hyp_str).split())}\n")
                rel_nosil_f.write(f"{utt_id} {' '.join(re.sub(r'sil', '', rel_str).split())}\n")
                can_nosil_f.write(f"{utt_id} {' '.join(re.sub(r'sil', '', can_str).split())}\n")

    hyp_f.close()
    rel_f.close()
    can_f.close()
    if hyp_nosil_f and rel_nosil_f and can_nosil_f:
        hyp_nosil_f.close()
        rel_nosil_f.close()
        can_nosil_f.close()


def main() -> None:
    print(f"I am process {os.getpid()}, running on {os.uname()[1]}: starting ({time.asctime()})")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--remove-special-token", action="store_true")
    parser.add_argument("--remove-sil", action="store_true")
    parser.add_argument("--special-token", type=str, default="<del>")
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--am", type=str, default="librispeech")
    parser.add_argument("--phn-dict", type=str, default="local/so762/vocab_merge.json")
    parser.add_argument("--model", type=str, default="HMamba")
    parser.add_argument("--model-conf", type=str, required=True)
    parser.add_argument("--gop-dir", type=str, required=True)
    parser.add_argument("--ssl-dir", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="best_mdd_f1_model.pth")
    args = parser.parse_args()

    conf = load_conf(args.model_conf)
    if args.model != "HMamba":
        raise ValueError(f"Invalid model {args.model}")

    audio_model = HMamba(**conf)
    te_dataset = GoPDataset(
        args.set,
        data_dir=args.gop_dir,
        data_dir2=args.ssl_dir,
        data_dir3=args.raw_dir,
        am=args.am,
        mode="mdd",
    )
    te_dataloader = DataLoader(cast(Dataset[tuple[object, ...]], te_dataset), batch_size=1, shuffle=False)
    recog(audio_model, te_dataloader, args)


if __name__ == "__main__":
    main()

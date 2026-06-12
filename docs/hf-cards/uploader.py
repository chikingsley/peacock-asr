"""Sequential HF dataset uploader — resumable, run inside tmux.

upload_large_folder tracks per-file state in <folder>/.cache/huggingface/, so re-running
after an interruption (or a reboot) skips already-uploaded bytes and finishes the commits.

  uv run --with huggingface_hub python docs/hf-cards/uploader.py
"""

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
CARDS = Path(__file__).resolve().parent

DATASETS = [
    (
        "Peacockery/persian-asr-corpus-v4",
        ROOT / "projects/persian-asr/src/finetune_omni/data/training/omnilingual/scribe-v4",
        None,  # README already on the Hub
    ),
    (
        "Peacockery/tajik-asr-corpus-v3",
        ROOT / "projects/tajik-asr/data/datasets/v3",
        CARDS / "ds-tajik-v3.md",
    ),
]


def main() -> None:
    api = HfApi()
    for repo_id, folder, card in DATASETS:
        print(f"=== DATASET {repo_id} <- {folder} ===", flush=True)
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        if card is not None:
            api.upload_file(
                path_or_fileobj=card, path_in_repo="README.md",
                repo_id=repo_id, repo_type="dataset",
            )
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="dataset")
        print(f"  DONE {repo_id}", flush=True)


if __name__ == "__main__":
    main()

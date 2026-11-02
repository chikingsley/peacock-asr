"""Stage and publish the measured October 2026 Tajik checkpoint."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from tajik_asr.parakeet import PROJECT

REPO_ID = "Peacockery/tajik-parakeet-tdt-110m-v1.1"
VERSION = "v1.1"
MODEL_FILENAME = "model.nemo"
PREVIOUS_REVISION = "1b213a53b8f8e39e553039f39200467dafe25a7f"
RUN_NAME = "tajik-parakeet-tdt-110m-upgraded-20261028"
MODEL_SHA256 = "9c4674e40c4c7d12087f2c5694cf6e9042cbc8afb7eab173935b87f5ee6cbdea"
REPO_ROOT = PROJECT.root.parents[1]
CARD = REPO_ROOT / "docs/hf-cards/tajik-parakeet-tdt-110m-v1.1"
RELEASE_DIR = PROJECT.data / "parakeet/releases" / f"{VERSION}-versioned-repository"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def portable_manifest(source: Path, destination: Path) -> dict:
    """Preserve audio identities and provenance with relative audio paths."""
    count, seconds = 0, 0.0
    with source.open() as rows, gzip.open(destination, "wt", encoding="utf-8") as output:
        for line in rows:
            row = json.loads(line)
            row["audio_filepath"] = f"audio/{Path(row['audio_filepath']).name}"
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            seconds += row["duration"]
    return {
        "file": f"manifests/{destination.name}",
        "rows": count,
        "hours": seconds / 3600,
        "source_manifest_sha256": digest(source),
        "published_sha256": digest(destination),
    }


def prepare() -> dict:
    selection = json.loads((PROJECT.runs / "upgrades-20261028/selection.json").read_text())
    model = PROJECT.runs / RUN_NAME / f"{RUN_NAME}_final.nemo"
    if selection["status"] != "complete" or Path(selection["model"]) != model:
        raise ValueError("Completed development selection must identify the release model")
    if digest(model) != MODEL_SHA256:
        raise ValueError("Release checkpoint hash differs from the evaluated artifact")
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    manifests = RELEASE_DIR / "manifests"
    manifests.mkdir(exist_ok=True)
    for name in ("README.md", "LICENSE", "NOTICE"):
        shutil.copyfile(CARD / name, RELEASE_DIR / name)
    shutil.copyfile(model, RELEASE_DIR / MODEL_FILENAME)
    tokenizer = RELEASE_DIR / "tokenizer"
    tokenizer.mkdir(exist_ok=True)
    with tarfile.open(model) as archive:
        for name in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
            members = [m for m in archive.getmembers() if m.name.endswith(f"_{name}")]
            if len(members) != 1:
                raise ValueError(f"Expected one bundled tokenizer artifact: {name}")
            handle = archive.extractfile(members[0])
            if handle is None:
                raise ValueError(f"Bundled tokenizer artifact is empty: {name}")
            (tokenizer / name).write_bytes(handle.read())
            if digest(tokenizer / name) != digest(PROJECT.default_tokenizer_dir / name):
                raise ValueError(f"Bundled tokenizer differs from the training input: {name}")
    sources = {
        "train": PROJECT.default_train_manifest,
        "dev": PROJECT.default_validation_manifest,
        "test": PROJECT.evaluation_manifests[0],
        **{
            name: PROJECT.runs / "clean-retrain-20261028" / f"{name}.jsonl"
            for name in ("fleurs600", "conversational")
        },
    }
    manifest_report = {
        name: portable_manifest(path, manifests / f"{name}.jsonl.gz")
        for name, path in sources.items()
    }
    prepared = json.loads((PROJECT.runs / "upgrades-20261028/plan.json").read_text())
    if manifest_report["train"]["source_manifest_sha256"] != prepared["input"]["sha256"]:
        raise ValueError("Training manifest differs from the evaluated run's prepared input")
    evaluation = {
        "selection": "lowest normalized development WER, followed by fixed held-out evaluation",
        "selected": selection["selected"],
        "inference_dtype": "fp32",
        "normalizer": "omni_curator.process.normalize with language tgk_Cyrl",
        "development": selection["development"],
        "heldout": {
            name: {key: score[key] for key in ("rows", "raw", "normalized")}
            for name, score in selection["heldout"].items()
        },
        "reference_types": {
            "fleurs600": "FLEURS human reference transcripts",
            "conversational": (
                "machine-generated transcripts; reference quality limits interpretation"
            ),
        },
        "previous_model": {
            "repo_id": "Peacockery/tajik-parakeet-tdt-110m-v1",
            "revision": "1b213a53b8f8e39e553039f39200467dafe25a7f",
            "fleurs600_normalized_wer_percent": 19.030451791056223,
            "conversational_recorded_wer_percent": 33.84549566067718,
            "conversational_scope": (
                "previous training included clips from reserved evaluation videos"
            ),
        },
    }
    save_json(RELEASE_DIR / "evaluation.json", evaluation)
    report = {
        "repository": REPO_ID,
        "release": VERSION,
        "artifact_id": "parakeet_tdt_110m_tajik_v3_step_100000",
        "model": {"file": MODEL_FILENAME, "sha256": MODEL_SHA256},
        "base_model": "nvidia/parakeet-tdt_ctc-110m",
        "base_model_sha256": "1e6b5c3bcaa390b479bd8112512ede515d2ef5b61495568f2c266d6d5e680c3a",
        "dataset": {
            "repository": "Peacockery/tajik-asr-corpus-v3",
            "revision": "3b05a4bb89104c21643081250729595347d1188e",
            "preparation": "v3-safe, duration and held-out source identity filtering",
        },
        "training": {
            "code_commit": "55af600a",
            "optimizer_steps": 100000,
            "schedule": "AdamW 3e-4, 2000 warmup steps, cosine decay to 1e-5",
            "initial_phase": {
                "steps": 14000,
                "loss_mode": "fused",
                "mixture": "original clip ratio",
            },
            "continuation": {
                "start_step": 14000,
                "restored_state": ["model", "optimizer", "scheduler", "global_step"],
                "loss_mode": "unfused",
                "precision": "bf16-mixed",
                "gold_clip_weight": 0.12,
                "pseudo_clip_weight": 0.88,
                "manifest_shard_rows": 200,
            },
            "tokenizer": f"fresh Tajik BPE-1024 from training text; embedded in {MODEL_FILENAME}",
        },
        "manifests": manifest_report,
        "files": {
            str(path.relative_to(RELEASE_DIR)): digest(path)
            for path in sorted(RELEASE_DIR.rglob("*"))
            if path.is_file() and path.name != "release.json"
        },
    }
    save_json(RELEASE_DIR / "release.json", report)
    return report


def publish(expected_parent: str) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    report = json.loads((RELEASE_DIR / "release.json").read_text())
    files = [RELEASE_DIR / name for name in report["files"]] + [RELEASE_DIR / "release.json"]
    for name, expected in report["files"].items():
        if digest(RELEASE_DIR / name) != expected:
            raise ValueError(f"Staged release changed after preparation: {name}")
    api = HfApi()
    current = api.model_info(REPO_ID, files_metadata=True)
    if current.sha != expected_parent:
        raise ValueError(
            "Repository changed after inspection; refresh the expected parent revision"
        )
    tags = {tag.name: tag.target_commit for tag in api.list_repo_refs(REPO_ID).tags}
    if VERSION in tags:
        raise FileExistsError(f"Release tag already exists: {VERSION}")
    previous = api.model_info("Peacockery/tajik-parakeet-tdt-110m-v1", revision=PREVIOUS_REVISION)
    if previous.sha != PREVIOUS_REVISION:
        raise ValueError("Original model revision differs from the recorded baseline")
    commit = api.create_commit(
        repo_id=REPO_ID,
        parent_commit=expected_parent,
        commit_message="Publish v1.1 model card, checkpoint provenance and evaluation",
        operations=[
            CommitOperationAdd(path_in_repo=str(p.relative_to(RELEASE_DIR)), path_or_fileobj=p)
            for p in files
        ],
    )
    api.create_tag(REPO_ID, tag=VERSION, revision=commit.oid)
    remote = api.model_info(REPO_ID, revision=VERSION, files_metadata=True)
    weights = next(file for file in remote.siblings if file.rfilename == MODEL_FILENAME)
    if remote.sha != commit.oid or weights.lfs.sha256 != MODEL_SHA256:
        raise ValueError("Published model or version differs from the staged release")
    for name in ("README.md", "LICENSE", "NOTICE", "release.json", "evaluation.json"):
        downloaded = Path(hf_hub_download(REPO_ID, name, revision=commit.oid, force_download=True))
        if digest(downloaded) != digest(RELEASE_DIR / name):
            raise ValueError(f"Published content differs from staging: {name}")
    result = {
        "repository": REPO_ID,
        "revision": commit.oid,
        "tag": VERSION,
        "url": f"https://huggingface.co/{REPO_ID}/tree/{VERSION}",
        "weight_sha256": MODEL_SHA256,
        "verified": (
            "remote LFS digest, tag revision, model card, license, notice, provenance and scores"
        ),
    }
    save_json(PROJECT.root / "docs/release-v1.1-publication.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "publish"])
    parser.add_argument("--expected-parent", help="Inspected Hub commit required for publication")
    args = parser.parse_args(argv)
    if args.action == "publish" and not args.expected_parent:
        parser.error("publish requires --expected-parent from the inspected repository")
    PROJECT.configure_environment()
    result = prepare() if args.action == "prepare" else publish(args.expected_parent)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

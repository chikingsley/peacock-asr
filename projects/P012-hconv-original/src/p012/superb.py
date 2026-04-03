from __future__ import annotations

from typing import TypeAlias

from .interfaces import InterfaceFeaturizer
from .vendor import ensure_third_party_on_path

ensure_third_party_on_path()

from s3prl.problem.asr.superb_pr import SuperbPR  # noqa: E402
from s3prl.problem.asv.superb_asv import SuperbASV  # noqa: E402
from s3prl.problem.common.superb_er import SuperbER  # noqa: E402
from s3prl.problem.common.superb_ic import SuperbIC  # noqa: E402


class InterfaceSuperbMixin:
    def default_config(self) -> dict:
        config = super().default_config()
        build_featurizer = dict(config.get("build_featurizer", {}))
        build_featurizer.setdefault("interface", "hconv")
        build_featurizer.setdefault("layer_selections", None)
        build_featurizer.setdefault("normalize", False)
        build_featurizer.setdefault("conv_kernel_size", 5)
        build_featurizer.setdefault("conv_kernel_stride", 3)
        build_featurizer.setdefault("output_dim", None)
        config["build_featurizer"] = build_featurizer
        return config

    def build_featurizer(self, build_featurizer: dict, upstream):
        return InterfaceFeaturizer(upstream, **build_featurizer)


class InterfaceSuperbPR(InterfaceSuperbMixin, SuperbPR):
    pass


class InterfaceSuperbER(InterfaceSuperbMixin, SuperbER):
    pass


class InterfaceSuperbIC(InterfaceSuperbMixin, SuperbIC):
    pass


class InterfaceSuperbASV(InterfaceSuperbMixin, SuperbASV):
    pass


ProblemType: TypeAlias = type[InterfaceSuperbPR | InterfaceSuperbER | InterfaceSuperbIC | InterfaceSuperbASV]

TASK_REGISTRY: dict[str, ProblemType] = {
    "pr": InterfaceSuperbPR,
    "er": InterfaceSuperbER,
    "ic": InterfaceSuperbIC,
    "asv": InterfaceSuperbASV,
    "sv": InterfaceSuperbASV,
}

TASK_DATASET_KEYS = {
    "pr": "dataset_root",
    "er": "iemocap",
    "ic": "dataset_root",
    "asv": "dataset_root",
    "sv": "dataset_root",
}


def get_problem(task: str):
    normalized = task.lower()
    if normalized not in TASK_REGISTRY:
        raise KeyError(f"Unsupported task '{task}'. Supported tasks: {', '.join(sorted(TASK_REGISTRY))}")
    return TASK_REGISTRY[normalized]()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OfferSearchSpec:
    gpu_name: str | None = None
    num_gpus: int = 1
    storage_gb: float = 150.0
    limit: int = 20
    order: str = "score-"
    offer_type: str = "on-demand"
    query_clauses: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LaunchInstanceSpec:
    gpu_name: str
    image: str
    num_gpus: int = 1
    disk_gb: float = 150.0
    limit: int = 3
    order: str = "score-"
    label: str | None = None
    region: str | None = None
    template_hash: str | None = None
    onstart_cmd: str | None = None
    env: tuple[str, ...] = ()
    query_clauses: tuple[str, ...] = ()
    login: str | None = None
    force: bool = False
    cancel_unavailable: bool = False


@dataclass(frozen=True, slots=True)
class TemplateSpec:
    name: str
    image: str
    disk_gb: float = 100.0
    description: str | None = None
    href: str | None = None
    repo: str | None = None
    env: str | None = None
    onstart_cmd: str | None = None
    search_params: str | None = None
    image_tag: str | None = None
    login: str | None = None
    ssh: bool = True
    direct: bool = True
    jupyter: bool = False
    public: bool = False


@dataclass(frozen=True, slots=True)
class OfferSummary:
    offer_id: int
    machine_id: int | None
    gpu_name: str
    num_gpus: int
    hourly_price: float | None
    gpu_ram_gb: float | None
    reliability: float | None
    geolocation: str | None
    score: float | None
    cuda_max_good: str | None
    driver_version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "offer_id": self.offer_id,
            "machine_id": self.machine_id,
            "gpu_name": self.gpu_name,
            "num_gpus": self.num_gpus,
            "hourly_price": self.hourly_price,
            "gpu_ram_gb": self.gpu_ram_gb,
            "reliability": self.reliability,
            "geolocation": self.geolocation,
            "score": self.score,
            "cuda_max_good": self.cuda_max_good,
            "driver_version": self.driver_version,
        }


@dataclass(frozen=True, slots=True)
class SSHConnection:
    user: str
    host: str
    port: int

    @property
    def uri(self) -> str:
        return f"ssh://{self.user}@{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "host": self.host,
            "port": self.port,
            "uri": self.uri,
        }


@dataclass(frozen=True, slots=True)
class VastInstanceSummary:
    instance_id: int
    machine_id: int | None
    actual_status: str | None
    label: str | None
    gpu_name: str | None
    num_gpus: int | None
    public_ipaddr: str | None
    ssh_connection: SSHConnection | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "machine_id": self.machine_id,
            "actual_status": self.actual_status,
            "label": self.label,
            "gpu_name": self.gpu_name,
            "num_gpus": self.num_gpus,
            "public_ipaddr": self.public_ipaddr,
            "ssh_connection": (
                self.ssh_connection.to_dict() if self.ssh_connection else None
            ),
        }


@dataclass(frozen=True, slots=True)
class LaunchResult:
    success: bool
    instance_id: int | None
    message: str | None
    error: str | None
    response: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "instance_id": self.instance_id,
            "message": self.message,
            "error": self.error,
            "response": self.response,
        }


@dataclass(frozen=True, slots=True)
class InstanceDestroyResult:
    destroyed: bool
    instance_id: int
    response: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "destroyed": self.destroyed,
            "instance_id": self.instance_id,
            "response": self.response,
        }


@dataclass(frozen=True, slots=True)
class TemplateSummary:
    template_id: int
    hash_id: str
    name: str
    image: str | None
    tag: str | None
    creator_id: int | None
    description: str | None
    recommended_disk_space_gb: float | None
    use_ssh: bool | None
    ssh_direct: bool | None
    jup_direct: bool | None
    private: bool | None
    count_created: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "hash_id": self.hash_id,
            "name": self.name,
            "image": self.image,
            "tag": self.tag,
            "creator_id": self.creator_id,
            "description": self.description,
            "recommended_disk_space_gb": self.recommended_disk_space_gb,
            "use_ssh": self.use_ssh,
            "ssh_direct": self.ssh_direct,
            "jup_direct": self.jup_direct,
            "private": self.private,
            "count_created": self.count_created,
        }


@dataclass(frozen=True, slots=True)
class TemplateUpsertResult:
    created: bool
    updated: bool
    template: TemplateSummary | None
    response: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "updated": self.updated,
            "template": None if self.template is None else self.template.to_dict(),
            "response": self.response,
        }


@dataclass(frozen=True, slots=True)
class TemplateDeleteResult:
    deleted: bool
    template: TemplateSummary | None
    response: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "deleted": self.deleted,
            "template": None if self.template is None else self.template.to_dict(),
            "response": self.response,
        }

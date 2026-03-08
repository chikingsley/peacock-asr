from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GpuSearchSpec:
    include_pricing: bool = False
    gpu_quantity: int = 1
    min_memory_gb: int | None = None
    max_memory_gb: int | None = None


@dataclass(frozen=True, slots=True)
class PodCreateSpec:
    name: str
    gpu_type_id: str | None = None
    image_name: str = ""
    cloud_type: str = "SECURE"
    support_public_ip: bool = False
    start_ssh: bool = True
    data_center_id: str | None = None
    country_code: str | None = None
    gpu_count: int = 1
    volume_in_gb: int = 0
    container_disk_in_gb: int | None = None
    min_vcpu_count: int = 1
    min_memory_in_gb: int = 1
    docker_args: str = ""
    ports: str | None = None
    volume_mount_path: str = "/workspace"
    env: dict[str, str] | None = None
    template_id: str | None = None
    network_volume_id: str | None = None
    allowed_cuda_versions: tuple[str, ...] = ()
    min_download: int | None = None
    min_upload: int | None = None
    instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunpodGpuSummary:
    gpu_id: str
    display_name: str
    memory_in_gb: int | None
    secure_cloud: bool | None = None
    community_cloud: bool | None = None
    secure_price: float | None = None
    community_price: float | None = None
    secure_spot_price: float | None = None
    community_spot_price: float | None = None
    minimum_bid_price: float | None = None
    uninterruptable_price: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id": self.gpu_id,
            "display_name": self.display_name,
            "memory_in_gb": self.memory_in_gb,
            "secure_cloud": self.secure_cloud,
            "community_cloud": self.community_cloud,
            "secure_price": self.secure_price,
            "community_price": self.community_price,
            "secure_spot_price": self.secure_spot_price,
            "community_spot_price": self.community_spot_price,
            "minimum_bid_price": self.minimum_bid_price,
            "uninterruptable_price": self.uninterruptable_price,
        }


@dataclass(frozen=True, slots=True)
class RunpodRuntimePort:
    ip: str | None
    is_ip_public: bool | None
    private_port: int | None
    public_port: int | None
    port_type: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ip": self.ip,
            "is_ip_public": self.is_ip_public,
            "private_port": self.private_port,
            "public_port": self.public_port,
            "port_type": self.port_type,
        }


@dataclass(frozen=True, slots=True)
class RunpodPodSummary:
    pod_id: str
    name: str | None
    desired_status: str | None
    cost_per_hr: float | None
    gpu_count: int | None
    machine_id: str | None
    gpu_display_name: str | None
    image_name: str | None
    container_disk_in_gb: int | None
    volume_in_gb: int | None
    volume_mount_path: str | None
    memory_in_gb: int | None
    vcpu_count: int | None
    runtime_ports: tuple[RunpodRuntimePort, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "name": self.name,
            "desired_status": self.desired_status,
            "cost_per_hr": self.cost_per_hr,
            "gpu_count": self.gpu_count,
            "machine_id": self.machine_id,
            "gpu_display_name": self.gpu_display_name,
            "image_name": self.image_name,
            "container_disk_in_gb": self.container_disk_in_gb,
            "volume_in_gb": self.volume_in_gb,
            "volume_mount_path": self.volume_mount_path,
            "memory_in_gb": self.memory_in_gb,
            "vcpu_count": self.vcpu_count,
            "runtime_ports": [port.to_dict() for port in self.runtime_ports],
        }


@dataclass(frozen=True, slots=True)
class RunpodUserSummary:
    user_id: str
    has_pub_key: bool
    network_volume_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "has_pub_key": self.has_pub_key,
            "network_volume_ids": list(self.network_volume_ids),
        }


@dataclass(frozen=True, slots=True)
class PodMutationResult:
    pod_id: str | None
    desired_status: str | None
    machine_id: str | None
    image_name: str | None
    pod_host_id: str | None
    terminated: bool | None
    response: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "pod_id": self.pod_id,
            "desired_status": self.desired_status,
            "machine_id": self.machine_id,
            "image_name": self.image_name,
            "pod_host_id": self.pod_host_id,
            "terminated": self.terminated,
            "response": self.response,
        }

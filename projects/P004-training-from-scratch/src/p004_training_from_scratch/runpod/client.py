from __future__ import annotations

import io
import tomllib
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Protocol, cast

import runpod

from p004_training_from_scratch.settings import ProjectSettings

from .models import (
    GpuSearchSpec,
    PodCreateSpec,
    PodMutationResult,
    RunpodGpuSummary,
    RunpodPodSummary,
    RunpodRuntimePort,
    RunpodUserSummary,
)

RUNPOD_CONFIG_PATH = Path.home() / ".runpod" / "config.toml"


class RunpodSDKProtocol(Protocol):
    api_key: str | None

    def get_gpus(self, api_key: str | None = None) -> list[dict[str, Any]]: ...

    def get_gpu(
        self,
        gpu_id: str,
        gpu_quantity: int = 1,
        api_key: str | None = None,
    ) -> dict[str, Any]: ...

    def get_pods(self, api_key: str | None = None) -> list[dict[str, Any]]: ...

    def get_pod(self, pod_id: str, api_key: str | None = None) -> dict[str, Any]: ...

    def get_user(self, api_key: str | None = None) -> dict[str, Any]: ...

    def create_pod(
        self,
        name: str,
        image_name: str = "",
        gpu_type_id: str | None = None,
        cloud_type: str = "ALL",
        support_public_ip: bool = True,
        start_ssh: bool = True,
        data_center_id: str | None = None,
        country_code: str | None = None,
        gpu_count: int = 1,
        volume_in_gb: int = 0,
        container_disk_in_gb: int | None = None,
        min_vcpu_count: int = 1,
        min_memory_in_gb: int = 1,
        docker_args: str = "",
        ports: str | None = None,
        volume_mount_path: str = "/runpod-volume",
        env: dict[str, str] | None = None,
        template_id: str | None = None,
        network_volume_id: str | None = None,
        allowed_cuda_versions: list[str] | None = None,
        min_download: int | None = None,
        min_upload: int | None = None,
        instance_id: str | None = None,
    ) -> dict[str, Any]: ...

    def stop_pod(self, pod_id: str) -> dict[str, Any]: ...

    def resume_pod(self, pod_id: str, gpu_count: int) -> dict[str, Any]: ...

    def terminate_pod(self, pod_id: str) -> Any: ...


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _mapping(value: Any, *, method_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        msg = f"{method_name} returned {type(value)!r}, expected mapping."
        raise TypeError(msg)
    return cast(Mapping[str, Any], value)


def _mapping_list(value: Any, *, method_name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        msg = f"{method_name} returned {type(value)!r}, expected list."
        raise TypeError(msg)
    rows: list[Mapping[str, Any]] = []
    for item in value:
        rows.append(_mapping(item, method_name=method_name))
    return rows


def _load_api_key_from_config(config_path: Path = RUNPOD_CONFIG_PATH) -> str | None:
    if not config_path.is_file():
        return None
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(data.get("apikey"), str):
        key = data["apikey"].strip()
        if key:
            return key
    default_profile = data.get("default")
    if isinstance(default_profile, Mapping):
        value = default_profile.get("apikey")
        if isinstance(value, str):
            key = value.strip()
            if key:
                return key
    return None


def _to_gpu_summary(row: Mapping[str, Any]) -> RunpodGpuSummary:
    lowest_price = row.get("lowestPrice")
    lowest_price_row = (
        cast(Mapping[str, Any], lowest_price)
        if isinstance(lowest_price, Mapping)
        else {}
    )
    return RunpodGpuSummary(
        gpu_id=str(row.get("id", "")),
        display_name=str(row.get("displayName", "")),
        memory_in_gb=_optional_int(row.get("memoryInGb")),
        secure_cloud=_optional_bool(row.get("secureCloud")),
        community_cloud=_optional_bool(row.get("communityCloud")),
        secure_price=_optional_float(row.get("securePrice")),
        community_price=_optional_float(row.get("communityPrice")),
        secure_spot_price=_optional_float(row.get("secureSpotPrice")),
        community_spot_price=_optional_float(row.get("communitySpotPrice")),
        minimum_bid_price=_optional_float(lowest_price_row.get("minimumBidPrice")),
        uninterruptable_price=_optional_float(
            lowest_price_row.get("uninterruptablePrice")
        ),
    )


def _to_runtime_port(row: Mapping[str, Any]) -> RunpodRuntimePort:
    return RunpodRuntimePort(
        ip=_optional_str(row.get("ip")),
        is_ip_public=_optional_bool(row.get("isIpPublic")),
        private_port=_optional_int(row.get("privatePort")),
        public_port=_optional_int(row.get("publicPort")),
        port_type=_optional_str(row.get("type")),
    )


def _to_pod_summary(row: Mapping[str, Any]) -> RunpodPodSummary:
    runtime = row.get("runtime")
    runtime_row = (
        cast(Mapping[str, Any], runtime) if isinstance(runtime, Mapping) else {}
    )
    ports = runtime_row.get("ports")
    runtime_ports = (
        tuple(
            _to_runtime_port(item)
            for item in _mapping_list(ports, method_name="ports")
        )
        if isinstance(ports, list)
        else ()
    )
    machine = row.get("machine")
    machine_row = (
        cast(Mapping[str, Any], machine) if isinstance(machine, Mapping) else {}
    )
    return RunpodPodSummary(
        pod_id=str(row.get("id", "")),
        name=_optional_str(row.get("name")),
        desired_status=_optional_str(row.get("desiredStatus")),
        cost_per_hr=_optional_float(row.get("costPerHr")),
        gpu_count=_optional_int(row.get("gpuCount")),
        machine_id=_optional_str(row.get("machineId")),
        gpu_display_name=_optional_str(machine_row.get("gpuDisplayName")),
        image_name=_optional_str(row.get("imageName")),
        container_disk_in_gb=_optional_int(row.get("containerDiskInGb")),
        volume_in_gb=_optional_int(row.get("volumeInGb")),
        volume_mount_path=_optional_str(row.get("volumeMountPath")),
        memory_in_gb=_optional_int(row.get("memoryInGb")),
        vcpu_count=_optional_int(row.get("vcpuCount")),
        runtime_ports=runtime_ports,
    )


def _to_pod_mutation_result(response: Any) -> PodMutationResult:
    if isinstance(response, Mapping):
        machine = response.get("machine")
        machine_row = (
            cast(Mapping[str, Any], machine) if isinstance(machine, Mapping) else {}
        )
        return PodMutationResult(
            pod_id=_optional_str(response.get("id")),
            desired_status=_optional_str(response.get("desiredStatus")),
            machine_id=_optional_str(response.get("machineId")),
            image_name=_optional_str(response.get("imageName")),
            pod_host_id=_optional_str(machine_row.get("podHostId")),
            terminated=None,
            response=dict(response),
        )
    return PodMutationResult(
        pod_id=None,
        desired_status=None,
        machine_id=None,
        image_name=None,
        pod_host_id=None,
        terminated=_optional_bool(response),
        response=response,
    )


class RunpodModuleAdapter:
    def __init__(self, sdk_module: RunpodSDKProtocol, *, api_key: str) -> None:
        self._sdk = sdk_module
        self._api_key = api_key

    def _call(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        self._sdk.api_key = self._api_key
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            return fn(*args, **kwargs)

    def get_gpus(self) -> list[dict[str, Any]]:
        return self._call(self._sdk.get_gpus, api_key=self._api_key)

    def get_gpu(self, gpu_id: str, *, gpu_quantity: int) -> dict[str, Any]:
        return self._call(
            self._sdk.get_gpu,
            gpu_id,
            gpu_quantity=gpu_quantity,
            api_key=self._api_key,
        )

    def get_pods(self) -> list[dict[str, Any]]:
        return self._call(self._sdk.get_pods, api_key=self._api_key)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._call(self._sdk.get_pod, pod_id, api_key=self._api_key)

    def get_user(self) -> dict[str, Any]:
        return self._call(self._sdk.get_user, api_key=self._api_key)

    def create_pod(self, spec: PodCreateSpec) -> dict[str, Any]:
        allowed_cuda_versions = (
            list(spec.allowed_cuda_versions) if spec.allowed_cuda_versions else None
        )
        return self._call(
            self._sdk.create_pod,
            spec.name,
            image_name=spec.image_name,
            gpu_type_id=spec.gpu_type_id,
            cloud_type=spec.cloud_type,
            support_public_ip=spec.support_public_ip,
            start_ssh=spec.start_ssh,
            data_center_id=spec.data_center_id,
            country_code=spec.country_code,
            gpu_count=spec.gpu_count,
            volume_in_gb=spec.volume_in_gb,
            container_disk_in_gb=spec.container_disk_in_gb,
            min_vcpu_count=spec.min_vcpu_count,
            min_memory_in_gb=spec.min_memory_in_gb,
            docker_args=spec.docker_args,
            ports=spec.ports,
            volume_mount_path=spec.volume_mount_path,
            env=spec.env,
            template_id=spec.template_id,
            network_volume_id=spec.network_volume_id,
            allowed_cuda_versions=allowed_cuda_versions,
            min_download=spec.min_download,
            min_upload=spec.min_upload,
            instance_id=spec.instance_id,
        )

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return self._call(self._sdk.stop_pod, pod_id)

    def resume_pod(self, pod_id: str, *, gpu_count: int) -> dict[str, Any]:
        return self._call(self._sdk.resume_pod, pod_id, gpu_count)

    def terminate_pod(self, pod_id: str) -> Any:
        return self._call(self._sdk.terminate_pod, pod_id)


class RunpodClient:
    def __init__(self, adapter: RunpodModuleAdapter) -> None:
        self._adapter = adapter

    @classmethod
    def from_env(cls) -> RunpodClient:
        settings = ProjectSettings.from_env()
        api_key = settings.runpod_api_key or _load_api_key_from_config()
        if api_key is None:
            msg = (
                "Missing RUNPOD_API_KEY and no usable "
                "~/.runpod/config.toml apikey found."
            )
            raise RuntimeError(msg)
        adapter = RunpodModuleAdapter(
            cast(RunpodSDKProtocol, runpod),
            api_key=api_key,
        )
        return cls(adapter)

    def get_user(self) -> RunpodUserSummary:
        response = _mapping(self._adapter.get_user(), method_name="get_user")
        network_volumes = response.get("networkVolumes")
        volume_rows = (
            _mapping_list(network_volumes, method_name="get_user.networkVolumes")
            if isinstance(network_volumes, list)
            else []
        )
        return RunpodUserSummary(
            user_id=str(response.get("id", "")),
            has_pub_key=_optional_str(response.get("pubKey")) is not None,
            network_volume_ids=tuple(
                str(volume.get("id", "")) for volume in volume_rows if volume.get("id")
            ),
        )

    def list_gpu_types(
        self,
        spec: GpuSearchSpec | None = None,
    ) -> list[RunpodGpuSummary]:
        resolved_spec = spec or GpuSearchSpec()
        rows = _mapping_list(self._adapter.get_gpus(), method_name="get_gpus")
        summaries: list[RunpodGpuSummary] = []
        for row in rows:
            summary = _to_gpu_summary(row)
            if resolved_spec.min_memory_gb is not None and (
                summary.memory_in_gb is None
                or summary.memory_in_gb < resolved_spec.min_memory_gb
            ):
                continue
            if resolved_spec.max_memory_gb is not None and (
                summary.memory_in_gb is not None
                and summary.memory_in_gb > resolved_spec.max_memory_gb
            ):
                continue
            if resolved_spec.include_pricing:
                pricing_row = _mapping(
                    self._adapter.get_gpu(
                        summary.gpu_id,
                        gpu_quantity=resolved_spec.gpu_quantity,
                    ),
                    method_name="get_gpu",
                )
                summary = _to_gpu_summary(pricing_row)
            summaries.append(summary)
        if resolved_spec.include_pricing:
            summaries.sort(
                key=lambda item: (
                    float("inf")
                    if item.uninterruptable_price is None
                    else item.uninterruptable_price,
                    item.display_name,
                )
            )
        else:
            summaries.sort(key=lambda item: item.display_name)
        return summaries

    def list_pods(self) -> list[RunpodPodSummary]:
        rows = _mapping_list(self._adapter.get_pods(), method_name="get_pods")
        return [_to_pod_summary(row) for row in rows]

    def get_pod(self, pod_id: str) -> RunpodPodSummary:
        row = _mapping(self._adapter.get_pod(pod_id), method_name="get_pod")
        return _to_pod_summary(row)

    def create_pod(self, spec: PodCreateSpec) -> PodMutationResult:
        response = self._adapter.create_pod(spec)
        return _to_pod_mutation_result(response)

    def stop_pod(self, pod_id: str) -> PodMutationResult:
        response = self._adapter.stop_pod(pod_id)
        return _to_pod_mutation_result(response)

    def resume_pod(self, pod_id: str, *, gpu_count: int) -> PodMutationResult:
        response = self._adapter.resume_pod(pod_id, gpu_count=gpu_count)
        return _to_pod_mutation_result(response)

    def terminate_pod(self, pod_id: str) -> PodMutationResult:
        response = self._adapter.terminate_pod(pod_id)
        return _to_pod_mutation_result(response)


__all__ = [
    "RUNPOD_CONFIG_PATH",
    "GpuSearchSpec",
    "RunpodClient",
    "RunpodSDKProtocol",
    "_load_api_key_from_config",
]

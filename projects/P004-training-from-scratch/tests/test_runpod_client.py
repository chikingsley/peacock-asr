from __future__ import annotations

from pathlib import Path
from typing import Any

from p004_training_from_scratch.runpod.client import (
    RUNPOD_CONFIG_PATH,
    RunpodClient,
    RunpodModuleAdapter,
    RunpodSDKProtocol,
    _load_api_key_from_config,
)
from p004_training_from_scratch.runpod.models import GpuSearchSpec, PodCreateSpec


class FakeRunpodSDK(RunpodSDKProtocol):
    def __init__(self) -> None:
        self.api_key: str | None = None
        self.last_create_kwargs: dict[str, Any] | None = None
        self.last_resume: tuple[str, int] | None = None

    def get_gpus(self, api_key: str | None = None) -> list[dict[str, Any]]:
        assert api_key == "test-key"
        return [
            {
                "id": "NVIDIA GeForce RTX 4090",
                "displayName": "RTX 4090",
                "memoryInGb": 24,
            },
            {
                "id": "NVIDIA H100 80GB HBM3",
                "displayName": "H100 SXM",
                "memoryInGb": 80,
            },
        ]

    def get_gpu(
        self,
        gpu_id: str,
        gpu_quantity: int = 1,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        assert api_key == "test-key"
        if gpu_id == "NVIDIA GeForce RTX 4090":
            return {
                "id": gpu_id,
                "displayName": "RTX 4090",
                "memoryInGb": 24,
                "secureCloud": True,
                "communityCloud": True,
                "securePrice": 0.59,
                "communityPrice": 0.39,
                "secureSpotPrice": 0.33,
                "communitySpotPrice": 0.29,
                "lowestPrice": {
                    "minimumBidPrice": 0.2,
                    "uninterruptablePrice": 0.39,
                },
            }
        return {
            "id": gpu_id,
            "displayName": "H100 SXM",
            "memoryInGb": 80,
            "secureCloud": True,
            "communityCloud": True,
            "securePrice": 2.49,
            "communityPrice": 1.99,
            "secureSpotPrice": 1.25,
            "communitySpotPrice": 1.05,
            "lowestPrice": {
                "minimumBidPrice": 0.9,
                "uninterruptablePrice": 1.99,
            },
        }

    def get_pods(self, api_key: str | None = None) -> list[dict[str, Any]]:
        assert api_key == "test-key"
        return [
            {
                "id": "pod-123",
                "name": "p004-train",
                "desiredStatus": "RUNNING",
                "costPerHr": 0.27,
                "gpuCount": 1,
                "machineId": "machine-7",
                "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
                "containerDiskInGb": 30,
                "volumeInGb": 500,
                "volumeMountPath": "/workspace",
                "memoryInGb": 125,
                "vcpuCount": 16,
                "runtime": {
                    "ports": [
                        {
                            "ip": "1.2.3.4",
                            "isIpPublic": True,
                            "privatePort": 22,
                            "publicPort": 32015,
                            "type": "tcp",
                        }
                    ]
                },
                "machine": {"gpuDisplayName": "RTX A5000"},
            }
        ]

    def get_pod(self, pod_id: str, api_key: str | None = None) -> dict[str, Any]:
        assert api_key == "test-key"
        return self.get_pods(api_key=api_key)[0] | {"id": pod_id}

    def get_user(self, api_key: str | None = None) -> dict[str, Any]:
        assert api_key == "test-key"
        return {
            "id": "user-1",
            "pubKey": "ssh-ed25519 AAAA",
            "networkVolumes": [{"id": "nv-1"}, {"id": "nv-2"}],
        }

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
    ) -> dict[str, Any]:
        print("raw_response: noisy")
        self.last_create_kwargs = {
            "name": name,
            "image_name": image_name,
            "gpu_type_id": gpu_type_id,
            "cloud_type": cloud_type,
            "support_public_ip": support_public_ip,
            "start_ssh": start_ssh,
            "gpu_count": gpu_count,
            "volume_in_gb": volume_in_gb,
            "container_disk_in_gb": container_disk_in_gb,
            "volume_mount_path": volume_mount_path,
            "env": env,
            "template_id": template_id,
            "network_volume_id": network_volume_id,
            "allowed_cuda_versions": allowed_cuda_versions,
            "instance_id": instance_id,
        }
        return {
            "id": "pod-new",
            "imageName": image_name,
            "machineId": "machine-9",
            "machine": {"podHostId": "host-44"},
        }

    def stop_pod(self, pod_id: str) -> dict[str, Any]:
        return {"id": pod_id, "desiredStatus": "EXITED"}

    def resume_pod(self, pod_id: str, gpu_count: int) -> dict[str, Any]:
        self.last_resume = (pod_id, gpu_count)
        return {"id": pod_id, "desiredStatus": "RUNNING", "machineId": "machine-8"}

    def terminate_pod(self, pod_id: str) -> bool:
        return pod_id == "pod-123"


def test_list_gpu_types_supports_pricing_enrichment() -> None:
    client = RunpodClient(RunpodModuleAdapter(FakeRunpodSDK(), api_key="test-key"))

    gpus = client.list_gpu_types(
        GpuSearchSpec(include_pricing=True, gpu_quantity=1, min_memory_gb=24)
    )

    assert [gpu.display_name for gpu in gpus] == ["RTX 4090", "H100 SXM"]
    assert gpus[0].uninterruptable_price == 0.39
    assert gpus[1].community_price == 1.99


def test_list_pods_normalizes_runtime_ports() -> None:
    client = RunpodClient(RunpodModuleAdapter(FakeRunpodSDK(), api_key="test-key"))

    pods = client.list_pods()

    assert len(pods) == 1
    assert pods[0].pod_id == "pod-123"
    assert pods[0].gpu_display_name == "RTX A5000"
    assert pods[0].runtime_ports[0].public_port == 32015


def test_create_pod_shapes_arguments_and_suppresses_sdk_stdout(
    capsys: Any,
) -> None:
    sdk = FakeRunpodSDK()
    client = RunpodClient(RunpodModuleAdapter(sdk, api_key="test-key"))

    result = client.create_pod(
        PodCreateSpec(
            name="p004-trainclean360",
            gpu_type_id="NVIDIA RTX 6000 Ada Generation",
            image_name="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
            cloud_type="SECURE",
            gpu_count=1,
            volume_in_gb=200,
            container_disk_in_gb=40,
            volume_mount_path="/workspace",
            env={"WANDB_MODE": "online"},
            template_id="runpod-torch-v280",
            allowed_cuda_versions=("12.8",),
        )
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert result.pod_id == "pod-new"
    assert sdk.last_create_kwargs == {
        "name": "p004-trainclean360",
        "image_name": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "gpu_type_id": "NVIDIA RTX 6000 Ada Generation",
        "cloud_type": "SECURE",
        "support_public_ip": False,
        "start_ssh": True,
        "gpu_count": 1,
        "volume_in_gb": 200,
        "container_disk_in_gb": 40,
        "volume_mount_path": "/workspace",
        "env": {"WANDB_MODE": "online"},
        "template_id": "runpod-torch-v280",
        "network_volume_id": None,
        "allowed_cuda_versions": ["12.8"],
        "instance_id": None,
    }


def test_resume_and_terminate_return_typed_results() -> None:
    sdk = FakeRunpodSDK()
    client = RunpodClient(RunpodModuleAdapter(sdk, api_key="test-key"))

    resume = client.resume_pod("pod-123", gpu_count=1)
    terminate = client.terminate_pod("pod-123")

    assert sdk.last_resume == ("pod-123", 1)
    assert resume.pod_id == "pod-123"
    assert resume.desired_status == "RUNNING"
    assert terminate.terminated is True


def test_get_user_normalizes_network_volumes() -> None:
    client = RunpodClient(RunpodModuleAdapter(FakeRunpodSDK(), api_key="test-key"))

    user = client.get_user()

    assert user.user_id == "user-1"
    assert user.has_pub_key is True
    assert user.network_volume_ids == ("nv-1", "nv-2")


def test_load_api_key_from_top_level_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('apikey = "abc123"\napiurl = "https://api.runpod.io"\n')

    assert _load_api_key_from_config(config_path) == "abc123"


def test_load_api_key_from_default_profile_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('[default]\napikey = "def456"\n')

    assert _load_api_key_from_config(config_path) == "def456"


def test_runpod_config_constant_points_to_user_config() -> None:
    assert RUNPOD_CONFIG_PATH.name == "config.toml"

from __future__ import annotations

from typing import Any

from p004_training_from_scratch.vast.client import (
    VastClient,
    VastSDKProtocol,
    _compute_ssh_connection,
)
from p004_training_from_scratch.vast.models import LaunchInstanceSpec, OfferSearchSpec


class FakeVastSDK(VastSDKProtocol):
    def __init__(self) -> None:
        self.last_search_kwargs: dict[str, Any] | None = None
        self.last_launch_kwargs: dict[str, Any] | None = None

    def search_offers(
        self,
        query: Any = None,
        *,
        type: Any = "on-demand",
        no_default: Any = False,
        new: Any = False,
        limit: int | None = None,
        disable_bundling: Any = False,
        storage: float = 5.0,
        order: str = "score-",
    ) -> list[dict[str, Any]]:
        del no_default, new, disable_bundling
        self.last_search_kwargs = {
            "query": query,
            "type": type,
            "limit": limit,
            "storage": storage,
            "order": order,
        }
        return [
            {
                "id": 101,
                "machine_id": 501,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "dph_total": 0.44,
                "gpu_ram": 24_576,
                "reliability": 0.98,
                "geolocation": "US, CA",
                "score": 312.4,
                "cuda_max_good": "12.8",
                "driver_version": "570.00",
            }
        ]

    def show_instances(self, *, quiet: Any = False) -> list[dict[str, Any]]:
        del quiet
        return [
            {
                "id": 42,
                "machine_id": 501,
                "actual_status": "running",
                "label": "p004-test",
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "public_ipaddr": "1.2.3.4",
                "ports": {"22/tcp": [{"HostPort": "41022"}]},
            }
        ]

    def launch_instance(
        self,
        *,
        gpu_name: str | None = None,
        num_gpus: str | None = None,
        region: str | None = None,
        image: Any = None,
        disk: float = 16.0,
        limit: int = 3,
        order: str = "score-",
        login: str | None = None,
        label: str | None = None,
        onstart_cmd: str | None = None,
        ssh: Any = False,
        extra: Any = None,
        env: str | None = None,
        force: Any = False,
        cancel_unavail: Any = False,
        template_hash: str | None = None,
        raw: bool = True,
    ) -> dict[str, Any]:
        self.last_launch_kwargs = {
            "gpu_name": gpu_name,
            "num_gpus": num_gpus,
            "region": region,
            "image": image,
            "disk": disk,
            "limit": limit,
            "order": order,
            "login": login,
            "label": label,
            "onstart_cmd": onstart_cmd,
            "ssh": ssh,
            "extra": extra,
            "env": env,
            "force": force,
            "cancel_unavail": cancel_unavail,
            "template_hash": template_hash,
            "raw": raw,
        }
        return {
            "success": True,
            "new_contract": 4242,
            "message": "Started",
            "error": None,
        }

    def destroy_instance(self, id: int | None = None) -> dict[str, Any]:
        return {"success": True, "destroyed": True, "id": id}


def test_search_offers_normalizes_rows() -> None:
    sdk = FakeVastSDK()
    client = VastClient(sdk)

    offers = client.search_offers(
        OfferSearchSpec(gpu_name="RTX 4090", query_clauses=("cpu_cores>=16",))
    )

    assert len(offers) == 1
    assert offers[0].offer_id == 101
    assert offers[0].hourly_price == 0.44
    assert offers[0].gpu_ram_gb == 24.0
    assert sdk.last_search_kwargs == {
        "query": 'gpu_name="RTX 4090" num_gpus=1 cpu_cores>=16',
        "type": "on-demand",
        "limit": 20,
        "storage": 150.0,
        "order": "score-",
    }


def test_show_instances_computes_ssh_from_port_binding() -> None:
    sdk = FakeVastSDK()
    client = VastClient(sdk)

    instances = client.show_instances()

    assert len(instances) == 1
    assert instances[0].ssh_connection is not None
    assert instances[0].ssh_connection.uri == "ssh://root@1.2.3.4:41022"


def test_compute_ssh_connection_falls_back_to_ssh_host() -> None:
    ssh = _compute_ssh_connection(
        {
            "ssh_host": "proxy.vast.ai",
            "ssh_port": 30000,
            "image_runtype": "jupyter",
        }
    )
    assert ssh is not None
    assert ssh.uri == "ssh://root@proxy.vast.ai:30001"


def test_launch_instance_formats_sdk_call() -> None:
    sdk = FakeVastSDK()
    client = VastClient(sdk)

    result = client.launch_instance(
        LaunchInstanceSpec(
            gpu_name="RTX 4090",
            image="nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
            label="p004-ref",
            template_hash="hash123",
            query_clauses=("cpu_cores>=16",),
            env=("WANDB_MODE=offline",),
        )
    )

    assert result.success is True
    assert result.instance_id == 4242
    assert sdk.last_launch_kwargs == {
        "gpu_name": "RTX_4090",
        "num_gpus": "1",
        "region": None,
        "image": "nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04",
        "disk": 150.0,
        "limit": 3,
        "order": "score-",
        "login": None,
        "label": "p004-ref",
        "onstart_cmd": None,
        "ssh": True,
        "extra": "cpu_cores>=16",
        "env": "-e WANDB_MODE=offline",
        "force": False,
        "cancel_unavail": False,
        "template_hash": "hash123",
        "raw": True,
    }


def test_destroy_instance_returns_typed_result() -> None:
    sdk = FakeVastSDK()
    client = VastClient(sdk)

    result = client.destroy_instance(instance_id=4242)

    assert result.destroyed is True
    assert result.instance_id == 4242
    assert result.response == {"success": True, "destroyed": True, "id": 4242}

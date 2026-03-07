from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import p004_training_from_scratch.vast.client as vast_client_module
from p004_training_from_scratch.vast.client import VastClient, VastSDKProtocol
from p004_training_from_scratch.vast.models import TemplateSpec


class FakeTemplateSDK:
    def __init__(self) -> None:
        self.last_output = "[]"
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[tuple[str | None, dict[str, Any]]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.template_row: dict[str, Any] | None = {
            "id": 111,
            "hash_id": "abc123",
            "name": "p004_ref_icefall_ssh",
            "image": "nvidia/cuda",
            "tag": "12.4.1-cudnn-devel-ubuntu22.04",
            "creator_id": 9,
            "desc": "template",
            "recommended_disk_space": 100.0,
            "use_ssh": True,
            "ssh_direct": True,
            "jup_direct": False,
            "private": True,
            "count_created": 0,
        }

    def search_templates(self, query: Any = None) -> Any:
        if self.template_row is None:
            self.last_output = "[]"
        elif query in {
            "name == p004_ref_icefall_ssh",
            "creator_id == 9",
        } and (self.create_calls or self.update_calls):
            self.last_output = json.dumps([self.template_row])
        else:
            self.last_output = "[]"
        return None

    def create_template(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        self.last_output = json.dumps({"success": True})
        return None

    def update_template(self, **kwargs: Any) -> Any:
        self.update_calls.append((kwargs.get("HASH_ID"), kwargs))
        self.last_output = json.dumps({"success": True})
        return None

    def delete_template(
        self,
        template_id: int | None = None,
        hash_id: str | None = None,
    ) -> Any:
        self.delete_calls.append({"template_id": template_id, "hash_id": hash_id})
        self.last_output = json.dumps({"success": True})
        self.template_row = None
        return None


@pytest.fixture
def isolated_template_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        vast_client_module,
        "TEMPLATE_REGISTRY",
        tmp_path / "vast_templates.json",
    )


def test_search_templates_parses_sdk_last_output(
    isolated_template_registry: None,
) -> None:
    sdk = FakeTemplateSDK()
    sdk.create_calls.append({"seed": True})
    client = VastClient(cast("VastSDKProtocol", sdk))
    templates = client.search_templates(query="name == p004_ref_icefall_ssh")
    assert len(templates) == 1
    assert templates[0].hash_id == "abc123"


def test_upsert_template_creates_when_missing(
    isolated_template_registry: None,
) -> None:
    sdk = FakeTemplateSDK()
    client = VastClient(cast("VastSDKProtocol", sdk))
    result = client.upsert_template(
        TemplateSpec(
            name="p004_ref_icefall_ssh",
            image="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            disk_gb=100.0,
        )
    )
    assert result.created is True
    assert result.updated is False
    assert len(sdk.create_calls) == 1
    assert result.template is not None


def test_upsert_template_updates_when_present(
    isolated_template_registry: None,
) -> None:
    sdk = FakeTemplateSDK()
    client = VastClient(cast("VastSDKProtocol", sdk))
    sdk.create_calls.append({"seed": True})
    result = client.upsert_template(
        TemplateSpec(
            name="p004_ref_icefall_ssh",
            image="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            disk_gb=100.0,
        )
    )
    assert result.created is False
    assert result.updated is True
    assert len(sdk.update_calls) == 1
    assert sdk.update_calls[0][0] == "abc123"


def test_delete_template_by_name_removes_registry_entry(
    isolated_template_registry: None,
) -> None:
    sdk = FakeTemplateSDK()
    client = VastClient(cast("VastSDKProtocol", sdk))
    result = client.upsert_template(
        TemplateSpec(
            name="p004_ref_icefall_ssh",
            image="nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            disk_gb=100.0,
        )
    )
    assert result.template is not None

    delete_result = client.delete_template(name="p004_ref_icefall_ssh")
    assert delete_result.deleted is True
    assert sdk.delete_calls == [{"template_id": 111, "hash_id": "abc123"}]
    registry_payload = json.loads(vast_client_module.TEMPLATE_REGISTRY.read_text())
    assert "p004_ref_icefall_ssh" not in registry_payload

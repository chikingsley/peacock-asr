from __future__ import annotations

import ast
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from dotenv import load_dotenv
from vastai import VastAI

from .models import (
    InstanceDestroyResult,
    LaunchInstanceSpec,
    LaunchResult,
    OfferSearchSpec,
    OfferSummary,
    SSHConnection,
    TemplateDeleteResult,
    TemplateSpec,
    TemplateSummary,
    TemplateUpsertResult,
    VastInstanceSummary,
    VolumeCreateResult,
    VolumeSummary,
)
from .query import build_docker_options, build_offer_query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_REGISTRY = Path(
    PROJECT_ROOT / "experiments/citrinet/control_plane/vast_templates.json"
)


class VastSDKProtocol(Protocol):
    last_output: str

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
    ) -> Any: ...

    def show_instances(self, *, quiet: Any = False) -> Any: ...

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
    ) -> Any: ...

    def destroy_instance(self, id: int | None = None) -> Any: ...

    def search_volumes(self, query: Any = None, *, storage: float = 1.0, order: str = "score-") -> Any: ...

    def show_volumes(self, type: str = "all") -> Any: ...

    def create_volume(
        self,
        id: int,
        size: float = 15,
        name: str | None = None,
    ) -> Any: ...

    def search_templates(self, query: Any = None) -> Any: ...

    def create_template(
        self,
        *,
        name: str | None = None,
        image: str | None = None,
        image_tag: str | None = None,
        href: str | None = None,
        repo: str | None = None,
        login: str | None = None,
        env: str | None = None,
        ssh: Any = False,
        jupyter: Any = False,
        direct: Any = False,
        onstart_cmd: str | None = None,
        search_params: str | None = None,
        disk_space: str | None = None,
        desc: str | None = None,
        public: Any = False,
    ) -> Any: ...

    def update_template(
        self,
        hash_id: str | None = None,
        *,
        name: str | None = None,
        image: str | None = None,
        image_tag: str | None = None,
        href: str | None = None,
        repo: str | None = None,
        login: str | None = None,
        env: str | None = None,
        ssh: Any = False,
        jupyter: Any = False,
        direct: Any = False,
        onstart_cmd: str | None = None,
        search_params: str | None = None,
        disk_space: str | None = None,
        desc: str | None = None,
        public: Any = False,
    ) -> Any: ...

    def delete_template(
        self,
        template_id: int | None = None,
        hash_id: str | None = None,
    ) -> Any: ...


def _require_api_key(env_var: str) -> str:
    value = os.environ.get(env_var)
    if value:
        return value
    msg = f"Missing required environment variable: {env_var}"
    raise RuntimeError(msg)


def _as_mapping_list(value: Any, *, method_name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        msg = f"{method_name} returned {type(value)!r}, expected list."
        raise TypeError(msg)
    mappings: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            msg = f"{method_name} item has type {type(item)!r}, expected mapping."
            raise TypeError(msg)
        mappings.append(item)
    return mappings


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_megabytes_to_gib(value: Any) -> float | None:
    amount = _optional_float(value)
    if amount is None:
        return None
    return amount / 1024.0


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _normalize_launch_gpu_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value.strip()).strip("_")


def _last_output_json(sdk: VastSDKProtocol, *, method_name: str) -> Any:
    raw = _optional_str(getattr(sdk, "last_output", None))
    if raw is None:
        msg = f"{method_name} produced no direct response and no parseable last_output."
        raise TypeError(msg)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"{method_name} emitted non-JSON last_output."
        raise TypeError(msg) from exc


def _last_output_mapping(
    sdk: VastSDKProtocol,
    *,
    method_name: str,
) -> dict[str, Any] | None:
    raw = _optional_str(getattr(sdk, "last_output", None))
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _prefix, sep, tail = raw.partition(":")
        if not sep:
            return None
        try:
            parsed = ast.literal_eval(tail.strip())
        except (SyntaxError, ValueError) as exc:
            msg = f"{method_name} emitted an unparseable mapping payload."
            raise TypeError(msg) from exc
        if not isinstance(parsed, dict):
            return None
        return {str(key): value for key, value in parsed.items()}
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in parsed.items()}


def _load_template_registry() -> dict[str, dict[str, Any]]:
    if not TEMPLATE_REGISTRY.is_file():
        return {}
    payload = json.loads(TEMPLATE_REGISTRY.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    registry: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            registry[key] = dict(value)
    return registry


def _save_template_registry(registry: Mapping[str, Mapping[str, Any]]) -> None:
    TEMPLATE_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: dict(value) for key, value in registry.items()}
    TEMPLATE_REGISTRY.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def _to_offer_summary(row: Mapping[str, Any]) -> OfferSummary:
    return OfferSummary(
        offer_id=int(row["id"]),
        machine_id=_optional_int(row.get("machine_id")),
        gpu_name=str(row.get("gpu_name", "")),
        num_gpus=int(row.get("num_gpus", 0)),
        hourly_price=_optional_float(row.get("dph_total")),
        gpu_ram_gb=_optional_megabytes_to_gib(row.get("gpu_ram")),
        reliability=_optional_float(row.get("reliability")),
        geolocation=_optional_str(row.get("geolocation")),
        score=_optional_float(row.get("score")),
        cuda_max_good=_optional_str(row.get("cuda_max_good")),
        driver_version=_optional_str(row.get("driver_version")),
    )


def _compute_ssh_connection(row: Mapping[str, Any]) -> SSHConnection | None:
    ports = row.get("ports")
    if isinstance(ports, Mapping):
        port_22 = ports.get("22/tcp")
        if isinstance(port_22, Sequence) and port_22:
            first_binding = port_22[0]
            if isinstance(first_binding, Mapping):
                host = _optional_str(row.get("public_ipaddr"))
                port = _optional_int(first_binding.get("HostPort"))
                if host and port:
                    return SSHConnection(user="root", host=host, port=port)

    ssh_host = _optional_str(row.get("ssh_host"))
    ssh_port = _optional_int(row.get("ssh_port"))
    image_runtype = _optional_str(row.get("image_runtype"))
    if ssh_host and ssh_port:
        if image_runtype and "jupyter" in image_runtype:
            ssh_port += 1
        return SSHConnection(user="root", host=ssh_host, port=ssh_port)
    return None


def _to_instance_summary(row: Mapping[str, Any]) -> VastInstanceSummary:
    return VastInstanceSummary(
        instance_id=int(row["id"]),
        machine_id=_optional_int(row.get("machine_id")),
        actual_status=_optional_str(row.get("actual_status")),
        label=_optional_str(row.get("label")),
        gpu_name=_optional_str(row.get("gpu_name")),
        num_gpus=_optional_int(row.get("num_gpus")),
        public_ipaddr=_optional_str(row.get("public_ipaddr")),
        ssh_connection=_compute_ssh_connection(row),
    )


def _to_volume_summary(row: Mapping[str, Any]) -> VolumeSummary:
    raw_id = row.get("id", row.get("volume_id"))
    if raw_id is None:
        msg = "volume row missing id"
        raise TypeError(msg)
    return VolumeSummary(
        volume_id=int(raw_id),
        name=_optional_str(row.get("name")),
        machine_id=_optional_int(row.get("machine_id")),
        size_gb=_optional_float(row.get("size")),
        status=_optional_str(row.get("status")),
    )


def _to_template_summary(row: Mapping[str, Any]) -> TemplateSummary:
    return TemplateSummary(
        template_id=int(row["id"]),
        hash_id=str(row["hash_id"]),
        name=str(row.get("name", "")),
        image=_optional_str(row.get("image")),
        tag=_optional_str(row.get("tag")),
        creator_id=_optional_int(row.get("creator_id")),
        description=_optional_str(row.get("desc")),
        recommended_disk_space_gb=_optional_float(row.get("recommended_disk_space")),
        use_ssh=_optional_bool(row.get("use_ssh")),
        ssh_direct=_optional_bool(row.get("ssh_direct")),
        jup_direct=_optional_bool(row.get("jup_direct")),
        private=_optional_bool(row.get("private")),
        count_created=_optional_int(row.get("count_created")),
    )


def _find_template_by_name(
    client: VastClient,
    *,
    name: str,
    creator_id: int | None,
) -> TemplateSummary | None:
    if creator_id is not None:
        creator_matches = client.search_templates(query=f"creator_id == {creator_id}")
        matched = next(
            (template for template in creator_matches if template.name == name),
            None,
        )
        if matched is not None:
            return matched
    matches = client.search_templates(query=f"name == {name}")
    return next((template for template in matches if template.name == name), None)


class VastClient:
    """Small, typed boundary around the dynamic Vast SDK."""

    def __init__(self, sdk: VastSDKProtocol) -> None:
        self._sdk = sdk

    @classmethod
    def from_env(cls, *, env_var: str = "VAST_API_KEY") -> VastClient:
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = _require_api_key(env_var)
        sdk = VastAI(api_key=api_key, raw=True, quiet=True)
        return cls(cast("VastSDKProtocol", sdk))

    def search_offers(self, spec: OfferSearchSpec) -> list[OfferSummary]:
        response = self._sdk.search_offers(
            query=build_offer_query(spec),
            type=spec.offer_type,
            limit=spec.limit,
            storage=spec.storage_gb,
            order=spec.order,
        )
        rows = _as_mapping_list(response, method_name="search_offers")
        return [_to_offer_summary(row) for row in rows]

    def search_templates(self, *, query: str | None = None) -> list[TemplateSummary]:
        response = self._sdk.search_templates(query=query)
        rows_raw = (
            response
            if response is not None
            else _last_output_json(
                self._sdk,
                method_name="search_templates",
            )
        )
        rows = _as_mapping_list(rows_raw, method_name="search_templates")
        return [_to_template_summary(row) for row in rows]

    def show_instances(self) -> list[VastInstanceSummary]:
        response = self._sdk.show_instances(quiet=False)
        rows = _as_mapping_list(response, method_name="show_instances")
        return [_to_instance_summary(row) for row in rows]

    def launch_instance(self, spec: LaunchInstanceSpec) -> LaunchResult:
        response = self._sdk.launch_instance(
            gpu_name=_normalize_launch_gpu_name(spec.gpu_name),
            num_gpus=str(spec.num_gpus),
            region=spec.region,
            image=spec.image,
            disk=spec.disk_gb,
            limit=spec.limit,
            order=spec.order,
            login=spec.login,
            label=spec.label,
            onstart_cmd=spec.onstart_cmd,
            ssh=True,
            extra=" ".join(spec.query_clauses) or None,
            env=build_docker_options(spec.env, spec.docker_options),
            force=spec.force,
            cancel_unavail=spec.cancel_unavailable,
            template_hash=spec.template_hash,
            raw=True,
        )
        if not isinstance(response, Mapping):
            msg = f"launch_instance returned {type(response)!r}, expected mapping."
            raise TypeError(msg)

        payload = dict(response)
        return LaunchResult(
            success=bool(payload.get("success")),
            instance_id=_optional_int(payload.get("new_contract")),
            message=_optional_str(payload.get("message")),
            error=_optional_str(payload.get("error")),
            response=payload,
        )

    def destroy_instance(self, *, instance_id: int) -> InstanceDestroyResult:
        response = self._sdk.destroy_instance(id=instance_id)
        payload: dict[str, Any] | None = None
        if isinstance(response, Mapping):
            payload = dict(response)
        elif response is None:
            parsed = _last_output_mapping(
                self._sdk,
                method_name="destroy_instance",
            )
            if parsed is not None:
                payload = parsed

        destroyed = bool(payload is None or payload.get("success", True))
        return InstanceDestroyResult(
            destroyed=destroyed,
            instance_id=instance_id,
            response=payload,
        )

    def search_volumes(self, *, query: str | None = None) -> list[VolumeSummary]:
        response = self._sdk.search_volumes(query=query, storage=1.0, order="score-")
        rows_raw = (
            response
            if response is not None
            else _last_output_json(self._sdk, method_name="search_volumes")
        )
        rows = _as_mapping_list(rows_raw, method_name="search_volumes")
        return [_to_volume_summary(row) for row in rows]

    def show_volumes(self, *, volume_type: str = "all") -> list[VolumeSummary]:
        response = self._sdk.show_volumes(type=volume_type)
        rows_raw = (
            response
            if response is not None
            else _last_output_json(self._sdk, method_name="show_volumes")
        )
        rows = _as_mapping_list(rows_raw, method_name="show_volumes")
        return [_to_volume_summary(row) for row in rows]

    def create_volume(
        self,
        *,
        instance_id: int,
        size_gb: float,
        name: str | None = None,
    ) -> VolumeCreateResult:
        response = self._sdk.create_volume(id=instance_id, size=size_gb, name=name)
        payload: dict[str, Any] | None = None
        if isinstance(response, Mapping):
            payload = dict(response)
        else:
            payload = _last_output_mapping(self._sdk, method_name="create_volume")

        volume = None
        if payload is not None:
            if isinstance(payload.get("volume"), Mapping):
                volume = _to_volume_summary(cast("Mapping[str, Any]", payload["volume"]))
            elif "id" in payload or "volume_id" in payload:
                volume = _to_volume_summary(payload)

        return VolumeCreateResult(
            created=bool(payload is None or payload.get("success", True)),
            source_instance_id=instance_id,
            volume=volume,
            response=payload,
        )

    def upsert_template(self, spec: TemplateSpec) -> TemplateUpsertResult:
        registry = _load_template_registry()
        registry_entry = registry.get(spec.name)
        existing_hash_id = None
        creator_id = None
        if registry_entry is not None:
            creator_id = _optional_int(registry_entry.get("creator_id"))
            existing_hash_id = _optional_str(registry_entry.get("hash_id"))
        if creator_id is not None:
            creator_matches = self.search_templates(query=f"creator_id == {creator_id}")
            current = next(
                (
                    template
                    for template in creator_matches
                    if template.name == spec.name
                ),
                None,
            )
            if current is not None:
                existing_hash_id = current.hash_id
        if existing_hash_id is None:
            matches = self.search_templates(query=f"name == {spec.name}")
            existing = next(
                (template for template in matches if template.name == spec.name),
                None,
            )
            if existing is not None:
                existing_hash_id = existing.hash_id

        if existing_hash_id is None:
            response = self._sdk.create_template(
                name=spec.name,
                image=spec.image,
                image_tag=spec.image_tag,
                href=spec.href,
                repo=spec.repo,
                login=spec.login,
                env=spec.env,
                ssh=spec.ssh,
                jupyter=spec.jupyter,
                direct=spec.direct,
                onstart_cmd=spec.onstart_cmd,
                search_params=spec.search_params,
                disk_space=str(spec.disk_gb),
                desc=spec.description,
                public=spec.public,
            )
            created = True
            updated = False
        else:
            response = cast(Any, self._sdk).update_template(
                HASH_ID=existing_hash_id,
                name=spec.name,
                image=spec.image,
                image_tag=spec.image_tag,
                href=spec.href,
                repo=spec.repo,
                login=spec.login,
                env=spec.env,
                ssh=spec.ssh,
                jupyter=spec.jupyter,
                direct=spec.direct,
                onstart_cmd=spec.onstart_cmd,
                search_params=spec.search_params,
                disk_space=str(spec.disk_gb),
                desc=spec.description,
                public=spec.public,
            )
            created = False
            updated = True

        response_payload: dict[str, Any] | None = None
        if isinstance(response, Mapping):
            response_payload = dict(response)
        else:
            try:
                response_payload = _last_output_mapping(
                    self._sdk,
                    method_name="template_mutation",
                )
            except TypeError:
                response_payload = None

        matched: TemplateSummary | None = None
        hash_id = (
            None
            if response_payload is None
            else _optional_str(response_payload.get("hash_id"))
        )
        if (
            response_payload is not None
            and "id" in response_payload
            and hash_id is not None
        ):
            matched = _to_template_summary(response_payload)
        elif existing_hash_id is None:
            refreshed = self.search_templates(query=f"name == {spec.name}")
            matched = next(
                (template for template in refreshed if template.name == spec.name),
                None,
            )
        elif existing_hash_id is not None:
            refreshed = self.search_templates(query=f"hash_id == {existing_hash_id}")
            matched = refreshed[0] if refreshed else None
            if matched is None and creator_id is not None:
                creator_matches = self.search_templates(
                    query=f"creator_id == {creator_id}"
                )
                matched = next(
                    (
                        template
                        for template in creator_matches
                        if template.name == spec.name
                    ),
                    None,
                )

        if matched is not None:
            registry[spec.name] = matched.to_dict()
            _save_template_registry(registry)

        return TemplateUpsertResult(
            created=created,
            updated=updated,
            template=matched,
            response=response_payload,
        )

    def delete_template(
        self,
        *,
        name: str | None = None,
        template_id: int | None = None,
        hash_id: str | None = None,
    ) -> TemplateDeleteResult:
        registry = _load_template_registry()
        matched: TemplateSummary | None = None
        creator_id: int | None = None

        if name is not None:
            registry_entry = registry.get(name)
            if registry_entry is not None:
                creator_id = _optional_int(registry_entry.get("creator_id"))
            matched = _find_template_by_name(self, name=name, creator_id=creator_id)
            if matched is not None:
                template_id = matched.template_id
                hash_id = matched.hash_id
                creator_id = matched.creator_id
            elif registry_entry is not None:
                if template_id is None:
                    template_id = _optional_int(registry_entry.get("template_id"))
                if hash_id is None:
                    hash_id = _optional_str(registry_entry.get("hash_id"))

        if template_id is None and hash_id is None:
            msg = (
                "delete_template requires at least one of: name, template_id, hash_id."
            )
            raise ValueError(msg)

        response = self._sdk.delete_template(template_id=template_id, hash_id=hash_id)

        response_payload: dict[str, Any] | None = None
        if isinstance(response, Mapping):
            response_payload = dict(response)
        else:
            try:
                response_payload = _last_output_mapping(
                    self._sdk,
                    method_name="delete_template",
                )
            except TypeError:
                response_payload = None

        deleted = True
        if name is not None:
            for attempt in range(5):
                refreshed = _find_template_by_name(
                    self,
                    name=name,
                    creator_id=creator_id,
                )
                if refreshed is None:
                    deleted = True
                    break
                deleted = False
                if attempt < 4:
                    time.sleep(1.0)

        if deleted:
            if name is not None:
                registry.pop(name, None)
            elif template_id is not None:
                registry = {
                    key: value
                    for key, value in registry.items()
                    if _optional_int(value.get("template_id")) != template_id
                }
            elif hash_id is not None:
                registry = {
                    key: value
                    for key, value in registry.items()
                    if _optional_str(value.get("hash_id")) != hash_id
                }
            _save_template_registry(registry)

        return TemplateDeleteResult(
            deleted=deleted,
            template=matched,
            response=response_payload,
        )


__all__ = [
    "VastClient",
    "_compute_ssh_connection",
]

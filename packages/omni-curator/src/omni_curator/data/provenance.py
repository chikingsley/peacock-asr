"""Typed provenance and license records stored inside ``Sample.meta``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast


@dataclass(frozen=True, slots=True)
class LicenseInfo:
    """Per-source license gate used by export and publication decisions."""

    id: str = "unknown"
    commercial_use: bool = False
    authority: str | None = None

    @classmethod
    def from_value(cls, value: LicenseValue | None) -> LicenseInfo:
        if value is None:
            return UNKNOWN_LICENSE
        if isinstance(value, LicenseInfo):
            return value
        if isinstance(value, tuple):
            license_id, commercial = value
            return cls(str(license_id), bool(commercial))
        license_id = value.get("id") or value.get("license") or "unknown"
        return cls(
            id=str(license_id),
            commercial_use=bool(value.get("commercial_use", False)),
            authority=str(value["authority"]) if value.get("authority") is not None else None,
        )

    def to_meta(self) -> dict[str, object]:
        out: dict[str, object] = {"id": self.id, "commercial_use": self.commercial_use}
        if self.authority is not None:
            out["authority"] = self.authority
        return out


type LicenseValue = LicenseInfo | tuple[str, bool] | Mapping[str, object]
UNKNOWN_LICENSE = LicenseInfo()


@dataclass(frozen=True, slots=True)
class TransformStep:
    """One operation applied to source audio/text before it entered the store."""

    operation: str
    tool: str
    version: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def to_meta(self) -> dict[str, object]:
        out: dict[str, object] = {"operation": self.operation, "tool": self.tool}
        if self.version is not None:
            out["version"] = self.version
        if self.parameters:
            out["parameters"] = dict(self.parameters)
        return out

    @classmethod
    def from_meta(cls, meta: Mapping[str, object]) -> TransformStep:
        params = meta.get("parameters")
        return cls(
            operation=str(meta.get("operation", "")),
            tool=str(meta.get("tool", "")),
            version=str(meta["version"]) if meta.get("version") is not None else None,
            parameters=cast("Mapping[str, object]", params) if isinstance(params, Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Where a sample came from and which policy controls it."""

    origin: str
    authority: str
    tool: str
    license: LicenseInfo = UNKNOWN_LICENSE
    transforms: tuple[TransformStep, ...] = ()

    def to_meta(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "authority": self.authority,
            "tool": self.tool,
            "license": self.license.to_meta(),
            "transforms": [step.to_meta() for step in self.transforms],
        }

    @classmethod
    def from_meta(cls, meta: Mapping[str, object]) -> SourceProvenance | None:
        raw = meta.get("provenance")
        if not isinstance(raw, Mapping):
            return None
        raw = cast("Mapping[str, object]", raw)
        raw_transforms = raw.get("transforms")
        transforms = (
            tuple(
                TransformStep.from_meta(cast("Mapping[str, object]", step))
                for step in raw_transforms
                if isinstance(step, Mapping)
            )
            if isinstance(raw_transforms, list)
            else ()
        )
        raw_license = raw.get("license")
        license_info = (
            LicenseInfo.from_value(cast("LicenseValue", raw_license))
            if isinstance(raw_license, Mapping | tuple | LicenseInfo)
            else UNKNOWN_LICENSE
        )
        return cls(
            origin=str(raw.get("origin", "")),
            authority=str(raw.get("authority", "")),
            tool=str(raw.get("tool", "")),
            license=license_info,
            transforms=transforms,
        )


def normalize_license_registry(
    licenses: Mapping[str, LicenseValue] | None,
) -> dict[str, LicenseInfo]:
    """Return a typed source -> license map, defaulting missing sources elsewhere to unknown."""
    if licenses is None:
        return {}
    return {source: LicenseInfo.from_value(value) for source, value in licenses.items()}

"""Tests for settings.py."""

from __future__ import annotations

from pathlib import Path


def test_settings_uses_project_local_env_file() -> None:
    from p011.settings import PROJECT_ROOT, Settings

    env_file = Settings.model_config["env_file"]
    assert isinstance(env_file, Path)
    assert env_file == PROJECT_ROOT / ".env"
    assert env_file.is_absolute()


def test_settings_env_prefix_is_p010() -> None:
    from p011.settings import Settings

    assert Settings.model_config["env_prefix"] == "P011_"


def test_ssl_interface_default_is_none() -> None:
    from p011.settings import Settings, SSLInterfaceMode

    field_info = Settings.model_fields["ssl_interface"]
    assert field_info.default is SSLInterfaceMode.NONE


def test_ssl_models_default_to_all_streams() -> None:
    from p011.settings import Settings
    from p011.ssl_features import SSL_MODEL_KEYS

    assert Settings(features_dir=Path("/tmp/p011-test-features")).ssl_models == SSL_MODEL_KEYS


def test_ssl_models_parse_and_canonicalize() -> None:
    from p011.settings import Settings

    settings = Settings(features_dir=Path("/tmp/p011-test-features"), ssl_models="wavlm,hubert")
    assert settings.ssl_models == ("hubert", "wavlm")
    assert settings.selected_ssl_dim == 2048
    assert settings.resolved_ssl_output_dim == 2048

from __future__ import annotations

from p004_training_from_scratch.nightly.attention_benchmark import (
    _build_comparison,
    _extract_flash_attn_output,
    _supports_fa4_capability,
)


def test_supports_fa4_capability_matches_current_runtime_contract() -> None:
    assert _supports_fa4_capability((9, 0)) is True
    assert _supports_fa4_capability((10, 0)) is True
    assert _supports_fa4_capability((11, 8)) is True
    assert _supports_fa4_capability((12, 0)) is False
    assert _supports_fa4_capability(None) is False


def test_build_comparison_handles_missing_flash_backend() -> None:
    payload = {
        "sdpa": {"forward_seconds": 0.01},
        "flash_attn_direct": {"ok": False},
        "fa4_expected_supported_on_device": False,
        "flex_attention_compiled": {
            "auto": {"ok": True, "mean_forward_seconds": 0.015},
            "triton": {"ok": True, "mean_forward_seconds": 0.008},
            "flash": {
                "ok": False,
                "error": {
                    "type": "AssertionError",
                    "message": "Unsupported compute capability.",
                },
            },
        },
    }
    comparison = _build_comparison(payload)
    assert comparison["auto_vs_sdpa_forward_ratio"] == 1.5
    assert comparison["triton_vs_sdpa_forward_ratio"] == 0.8
    assert comparison["flash_vs_sdpa_forward_ratio"] is None
    assert comparison["flash_backend_supported"] is False
    assert comparison["flash_backend_expected_supported"] is False
    assert comparison["direct_flash_attn_ok"] is False


def test_extract_flash_attn_output_accepts_tuple_payload() -> None:
    tensor = object()

    extracted = _extract_flash_attn_output((tensor, {"aux": True}))

    assert extracted is tensor

import json
from pathlib import Path

from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.coreml.plan import (
    CoreMLPlanOptions,
    build_coreml_plan,
    feature_frames_for_seconds,
    moss_audio_tokens_for_frames,
)


def test_coreml_audio_token_math_matches_reference_fixture() -> None:
    assert moss_audio_tokens_for_frames(1484) == 193
    assert feature_frames_for_seconds(seconds=30.0, sample_rate=16_000, hop_length=160) == 3000
    assert moss_audio_tokens_for_frames(3000) == 390


def test_coreml_plan_uses_moss_shapes() -> None:
    config = MossModelConfig.from_moss_dict(
        {
            "audio_config": {
                "encoder_layers": 32,
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "output_dim": 2048,
            },
            "language_config": {
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 151936,
                "rope_theta": 1_000_000,
            },
            "adapter_hidden_size": 8192,
        }
    )
    plan = build_coreml_plan(
        config=config,
        config_path=Path("config.json"),
        options=CoreMLPlanOptions(),
    )

    assert plan["model"]["text_decoder"]["layers"] == 28
    assert plan["model"]["text_decoder"]["hidden_size"] == 2048
    assert plan["model"]["audio_encoder"]["layers"] == 32
    assert plan["derived_shapes"]["max_audio_tokens"] == 390
    assert plan["derived_shapes"]["max_prefill_prompt_tokens"] == 400
    assert plan["derived_shapes"]["prefill_margin_tokens"] == 112
    assert plan["derived_shapes"]["padded_cache_len"] == 768
    assert plan["derived_shapes"]["kv_cache"]["shape_per_layer"] == [1, 8, 768, 128]
    assert plan["derived_shapes"]["kv_cache"]["total_mib_fp16"] == 84.0
    assert not plan["warnings"]


def test_coreml_bundle_manifest_declares_required_cache_presets() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((project_root / "runtime/moss_bundle_manifest.json").read_text())

    assert manifest["version"] == 1
    assert manifest["default_cache_preset"] == "compat-768"
    artifacts = manifest["artifacts"]
    assert artifacts["token_package_path"] == "compiled/moss_token_embedding.mlmodelc"
    assert artifacts["tokenizer_path"] == "../../artifacts/coreml/moss_tokenizer.json"

    presets = {item["name"]: item for item in manifest["cache_presets"]}
    assert set(presets) == {"short-512", "compat-768", "matched-768"}
    assert presets["short-512"]["cache_len"] == 512
    assert presets["short-512"]["step_package_path"].endswith(
        "moss_decoder_step_padded_512.mlmodelc"
    )
    assert presets["compat-768"]["cache_len"] == 768
    assert presets["matched-768"]["status"] == "experimental-mpsgraph-blocked"

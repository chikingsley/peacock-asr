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

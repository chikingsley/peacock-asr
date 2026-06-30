from typing import cast

from moss_mlx_conversion import DEFAULT_MODEL_ID
from moss_mlx_conversion.backend import MossTranscribeBackend, STTOutput
from moss_mlx_conversion.backend.serving import MossSerialAdapter, TranscriptionRequest
from moss_mlx_conversion.config import MossModelConfig
from moss_mlx_conversion.conversion.convert import convert_tensor
from moss_mlx_conversion.conversion.weights import map_source_key
from moss_mlx_conversion.runtime.eval import normalize_for_wer
from moss_mlx_conversion.runtime.quantization import applies_to_scope


def test_default_model_id() -> None:
    assert DEFAULT_MODEL_ID == "OpenMOSS-Team/MOSS-Transcribe-preview-2B"


def test_weight_key_mapping() -> None:
    assert (
        map_source_key("model.language_model.layers.0.self_attn.q_proj.weight")
        == "model.layers.0.self_attn.q_proj.weight"
    )
    assert map_source_key("model.audio_model.conv2d1.weight") == "audio_model.conv2d1.weight"
    assert (
        map_source_key("model.audio_adapter.gate_proj.weight")
        == "audio_adapter.gate_proj.weight"
    )


def test_mlx_config_round_trips_converted_text_config() -> None:
    config = MossModelConfig.from_moss_dict(
        {
            "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "rope_theta": 1_000_000,
            },
            "audio_config": {"output_dim": 2048},
            "end_token_id": 151645,
        }
    )

    assert config.text_config.hidden_size == 2048
    assert config.text_config.num_hidden_layers == 28
    assert config.text_config.rope_theta == 1_000_000
    assert config.audio_config.output_dim == 2048
    assert config.end_token_id == 151645


def test_conv2d_weight_conversion_uses_mlx_layout() -> None:
    import torch

    tensor = torch.arange(2 * 1 * 3 * 3, dtype=torch.bfloat16).reshape(2, 1, 3, 3)
    converted = convert_tensor("model.audio_model.conv2d1.weight", tensor, dtype="bf16")

    assert list(converted.shape) == [2, 3, 3, 1]


def test_normalize_for_wer_removes_case_and_punctuation() -> None:
    assert normalize_for_wer("With her white paint, and her scarlet smokestack.") == (
        "with her white paint and her scarlet smokestack"
    )


def test_backend_output_contract() -> None:
    output = STTOutput(text="hello", prompt_tokens=3, generation_tokens=2)
    assert output.text == "hello"
    assert output.segments == []
    assert output.language == "English"


def test_serving_adapter_is_serial() -> None:
    adapter = MossSerialAdapter(backend=cast("MossTranscribeBackend", object()))
    request = TranscriptionRequest(audio="audio.wav")
    assert adapter.supports_batch(request) is False
    assert adapter.max_batch_size == 1


def test_backend_factory_exists_without_loading_weights() -> None:
    assert callable(MossTranscribeBackend.from_pretrained)


def test_quantization_scope_predicates() -> None:
    assert applies_to_scope("model.layers.0.self_attn.q_proj", "text-decoder")
    assert not applies_to_scope("audio_model.layers.0.fc1", "text-decoder")
    assert applies_to_scope("audio_adapter.gate_proj", "text-and-adapter")

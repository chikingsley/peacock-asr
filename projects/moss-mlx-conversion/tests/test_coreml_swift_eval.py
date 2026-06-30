import argparse
from pathlib import Path

from moss_mlx_conversion.coreml.swift_eval import safe_stem, swift_command
from moss_mlx_conversion.runtime.eval import StreamingExample


def test_safe_stem_uses_row_and_sanitized_id() -> None:
    example = StreamingExample(
        row_idx=7,
        example_id="abc/def ghi",
        reference="hello",
        audio_url="https://example.test/audio.flac",
    )

    assert safe_stem(example) == "000007-abc_def_ghi"


def test_swift_command_uses_absolute_runtime_inputs() -> None:
    args = argparse.Namespace(
        swift_package_path=Path("swift/MossCoreMLFixture"),
        packages_dir=Path("coreml/build"),
        fixture=Path("artifacts/coreml/moss_swift_fixture_compact.json"),
        audio_max_frames=3000,
        audio_package="compiled_audio_30s/moss_audio_encoder_adapter_30s_padded.mlmodelc",
        compute_units="cpu-gpu",
        max_new_tokens=160,
    )

    command = swift_command(
        project_root=Path("/example/moss"),
        args=args,
        audio_path=Path("/example/audio.wav"),
        reference_path=Path("/example/reference.txt"),
        output_path=Path("/example/out.json"),
    )

    assert command[:6] == [
        "swift",
        "run",
        "--package-path",
        "/example/moss/swift/MossCoreMLFixture",
        "-c",
        "release",
    ]
    assert "--audio-max-frames" in command
    assert "3000" in command
    assert "--reference-text-file" in command
    assert "/example/reference.txt" in command

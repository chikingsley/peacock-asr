import pytest

from omni_curator.create.fuse import extract_transcript


@pytest.mark.parametrize(
    "response",
    [
        "I need to analyze this hypothesis and determine the correct Tajik transcript.",
        "<transcript>partial",
        "<transcript></transcript>",
        "<transcript>one</transcript><transcript>two</transcript>",
    ],
)
def test_malformed_fusion_output_is_rejected(response):
    with pytest.raises(ValueError, match="exactly one nonempty"):
        extract_transcript(response)


def test_reasoning_outside_transcript_stays_out_of_label():
    assert extract_transcript("Reasoning here. <transcript>Салом</transcript>") == "Салом"

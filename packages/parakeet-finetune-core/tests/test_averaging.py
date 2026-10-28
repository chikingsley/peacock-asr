import pytest

from parakeet_finetune_core.averaging import average_states

torch = pytest.importorskip("torch")


def test_average_preserves_precision_and_best_integer_buffers():
    first = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(7)}
    second = {"weight": torch.tensor([3.0, 5.0]), "count": torch.tensor(9)}
    result = average_states([first, second])
    assert torch.equal(result["weight"], torch.tensor([2.0, 4.0]))
    assert result["weight"].dtype == first["weight"].dtype
    assert result["count"].item() == 7
    assert first["weight"].tolist() == [1.0, 3.0]


@pytest.mark.parametrize(
    "second",
    [
        {"other": torch.tensor([1.0])},
        {"weight": torch.tensor([1.0, 2.0])},
        {"weight": torch.tensor([float("nan")])},
        {"weight": torch.tensor([1.0], dtype=torch.bfloat16)},
    ],
)
def test_incompatible_or_nonfinite_states_fail(second):
    with pytest.raises(ValueError, match="Checkpoint"):
        average_states([{"weight": torch.tensor([1.0])}, second])


def test_single_checkpoint_requires_a_real_average():
    with pytest.raises(ValueError, match="at least two"):
        average_states([{"weight": torch.tensor([1.0])}])

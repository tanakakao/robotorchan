import pytest
import torch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction

from robotorchan.acquisition.mixed_one_shot import optimize_mixed_one_shot_acqf


class _CrossCategoryOneShot(OneShotAcquisitionFunction):
    def __init__(self) -> None:
        super().__init__(model=None)

    def get_augmented_q_batch_size(self, q: int) -> int:
        return q + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actual = x[..., 0, :]
        fantasy = x[..., 1, :]
        continuous_score = -((actual[..., 0] - 0.25) ** 2)
        continuous_score -= (fantasy[..., 0] - 0.75) ** 2
        category_score = (actual[..., 1] - fantasy[..., 1]).abs()
        return continuous_score + category_score

    def extract_candidates(self, x_full: torch.Tensor) -> torch.Tensor:
        return x_full[..., :1, :]


def test_mixed_one_shot_optimizer_allows_cross_category_fantasy() -> None:
    acquisition = _CrossCategoryOneShot()
    candidate, value = optimize_mixed_one_shot_acqf(
        acquisition,
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
        categorical_features={1: [0.0, 1.0]},
        num_restarts=2,
        raw_samples=16,
        max_assignments=4,
    )

    assert candidate.shape == torch.Size([1, 2])
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert value.item() > 0.9


def test_mixed_one_shot_optimizer_rejects_large_assignment_space() -> None:
    acquisition = _CrossCategoryOneShot()

    try:
        optimize_mixed_one_shot_acqf(
            acquisition,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            categorical_features={1: [0.0, 1.0]},
            max_assignments=3,
        )
    except ValueError as error:
        assert "assignment space" in str(error)
    else:
        raise AssertionError("expected assignment-space guard")


@pytest.mark.parametrize(
    ("categorical_features", "match"),
    [
        ({-1: [0.0, 1.0]}, "outside"),
        ({2: [0.0, 1.0]}, "outside"),
        ({1: []}, "no candidate values"),
        ({1: [-1.0, 0.0]}, "outside bounds"),
        ({1: [0.0, 2.0]}, "outside bounds"),
    ],
)
def test_mixed_one_shot_optimizer_validates_categorical_domain(
    categorical_features: dict[int, list[float]],
    match: str,
) -> None:
    acquisition = _CrossCategoryOneShot()

    with pytest.raises(ValueError, match=match):
        optimize_mixed_one_shot_acqf(
            acquisition,
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double),
            categorical_features=categorical_features,
            max_assignments=4,
        )

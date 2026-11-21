"""Independent finite-sample multiclass qBALD information-gain reference."""

import itertools
import math

import torch
from botorch.models.model import Model

from robotorchan.acquisition import qBALD


class FixedClassProbabilityModel(Model):
    num_classes = 3

    def __init__(self, samples: torch.Tensor) -> None:
        super().__init__()
        self.samples = samples

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: torch.Tensor, **kwargs: object) -> object:
        raise NotImplementedError("qBALD uses sample_class_probabilities, not posterior.")

    def mutual_information(self, X: torch.Tensor, *, num_samples: int = 128) -> torch.Tensor:
        raise NotImplementedError("qBALD uses joint probability samples, not marginal MI.")

    def sample_class_probabilities(
        self, X: torch.Tensor, sample_shape: torch.Size | None = None
    ) -> torch.Tensor:
        assert sample_shape == torch.Size([self.samples.shape[0]])
        assert X.shape[-2:] == (*self.samples.shape[-2:-1], 1)
        return self.samples


def test_multiclass_qbald_matches_explicit_joint_label_reference() -> None:
    # Three posterior draws, two correlated-through-posterior query locations.
    samples = torch.tensor(
        [
            [[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]],
            [[[0.2, 0.7, 0.1], [0.6, 0.2, 0.2]]],
            [[[0.1, 0.2, 0.7], [0.2, 0.2, 0.6]]],
        ],
        dtype=torch.double,
    )
    model = FixedClassProbabilityModel(samples)
    X = torch.tensor([[[0.2], [0.8]]], dtype=torch.double)
    actual = qBALD(model, num_samples=3)(X)

    # Independent reference: enumerate every possible pair of class labels.
    predictive_entropy = 0.0
    conditional_entropy = 0.0
    for first, second in itertools.product(range(3), repeat=2):
        probabilities = [
            samples[draw, 0, 0, first].item() * samples[draw, 0, 1, second].item()
            for draw in range(3)
        ]
        predictive_probability = sum(probabilities) / 3
        predictive_entropy -= predictive_probability * math.log(predictive_probability)
        conditional_entropy -= sum(p * math.log(p) for p in probabilities) / 3

    expected = torch.tensor([predictive_entropy - conditional_entropy], dtype=torch.double)
    assert actual.shape == (1,)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual.item() > 0.0

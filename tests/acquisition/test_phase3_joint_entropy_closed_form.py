"""Independent closed-form Gaussian entropy references for correlated q=2 batches."""

import math
from types import SimpleNamespace

import torch
from botorch.models.model import Model

from robotorchan.acquisition import qJointPosteriorEntropy


class FixedCovarianceModel(Model):
    def __init__(self, covariance: torch.Tensor) -> None:
        super().__init__()
        self.covariance = covariance

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: torch.Tensor, **kwargs: object) -> object:
        assert X.shape[-2] == 2
        return SimpleNamespace(mvn=SimpleNamespace(covariance_matrix=self.covariance))


def test_joint_entropy_matches_closed_form_bivariate_gaussian() -> None:
    # Same marginal variances, but distinct within-batch correlations.
    # The independent scalar reference uses det(Sigma) = v1*v2 - cov**2.
    variances = [(2.0, 3.0, 0.0), (2.0, 3.0, 1.5), (2.0, 3.0, -1.5)]
    matrices = torch.tensor(
        [[[v1, cov], [cov, v2]] for v1, v2, cov in variances],
        dtype=torch.double,
    )
    X = torch.zeros(3, 2, 1, dtype=torch.double)
    jitter = 1e-8
    actual = qJointPosteriorEntropy(FixedCovarianceModel(matrices), jitter=jitter)(X)

    expected = torch.tensor(
        [
            math.log(2.0 * math.pi * math.e)
            + 0.5 * math.log((v1 + jitter) * (v2 + jitter) - cov**2)
            for v1, v2, cov in variances
        ],
        dtype=torch.double,
    )
    assert actual.shape == (3,)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert actual[0] > actual[1]
    torch.testing.assert_close(actual[1], actual[2], atol=1e-12, rtol=1e-12)

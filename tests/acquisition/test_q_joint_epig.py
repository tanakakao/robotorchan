"""Joint Gaussian EPIG contracts."""

from types import SimpleNamespace

import pytest
import torch
from botorch.models.model import Model

from robotorchan.acquisition.active_learning.epig import qJointExpectedPredictiveInformationGain


@pytest.mark.parametrize("q", [1, 2, 4])
def test_joint_epig_uses_full_candidate_covariance(q: int) -> None:
    class CovarianceModel(Model):
        @property
        def num_outputs(self) -> int:
            return 1

        def posterior(self, X: torch.Tensor, **kwargs: object) -> object:
            covariance = torch.eye(q + 1, dtype=torch.double)
            covariance[:q, q] = 0.2
            covariance[q, :q] = 0.2
            matrix = covariance.expand(*X.shape[:-2], q + 1, q + 1)
            return SimpleNamespace(mvn=SimpleNamespace(covariance_matrix=matrix))

    model = CovarianceModel()
    target_X = torch.tensor([[0.5]], dtype=torch.double)
    X = torch.linspace(0.0, 1.0, q, dtype=torch.double).reshape(1, q, 1)
    acquisition = qJointExpectedPredictiveInformationGain(model, target_X, observation_noise=0.1)
    actual = acquisition(X)
    expected = -0.5 * torch.log1p(torch.tensor(-q * 0.04 / 1.21, dtype=torch.double))
    assert actual.shape == torch.Size([1])
    torch.testing.assert_close(actual, expected.reshape(1))

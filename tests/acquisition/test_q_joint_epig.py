"""Joint Gaussian EPIG contracts."""

from unittest.mock import Mock

import pytest
import torch

from robotorchan.acquisition.active_learning.epig import qJointExpectedPredictiveInformationGain


@pytest.mark.parametrize("q", [1, 2, 4])
def test_joint_epig_uses_full_candidate_covariance(q: int) -> None:
    model = Mock()
    model.num_outputs = 1
    model._is_ensemble = False

    def posterior(X: torch.Tensor) -> Mock:
        covariance = torch.eye(q + 1, dtype=torch.double)
        covariance[:q, q] = 0.2
        covariance[q, :q] = 0.2
        result = Mock()
        result.mvn.covariance_matrix = covariance.expand(*X.shape[:-2], q + 1, q + 1)
        return result

    model.posterior.side_effect = posterior
    target_X = torch.tensor([[0.5]], dtype=torch.double)
    X = torch.linspace(0.0, 1.0, q, dtype=torch.double).reshape(1, q, 1)
    acquisition = qJointExpectedPredictiveInformationGain(
        model, target_X, observation_noise=0.1
    )
    actual = acquisition(X)
    expected = -0.5 * torch.log1p(torch.tensor(-q * 0.04 / 1.21))
    assert actual.shape == torch.Size([1])
    torch.testing.assert_close(actual, expected.reshape(1))

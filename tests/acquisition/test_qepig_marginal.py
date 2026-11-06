"""Marginal qEPIG aggregation contracts."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.acquisition import ExpectedPredictiveInformationGain
from robotorchan.acquisition.active_learning.epig import qExpectedPredictiveInformationGain


@pytest.mark.parametrize("q", [1, 2, 4])
def test_qepig_averages_pointwise_epig(q: int) -> None:
    model = torch.nn.Linear(1, 1)
    X = torch.arange(q, dtype=torch.double).reshape(1, q, 1)
    target = torch.zeros(2, 1, dtype=torch.double)
    expected_marginals = torch.arange(1, q + 1, dtype=torch.double).reshape(1, q)
    with patch.object(
        ExpectedPredictiveInformationGain, "forward", return_value=expected_marginals
    ) as forward:
        actual = qExpectedPredictiveInformationGain(model, target)(X)
    assert actual.shape == torch.Size([1])
    torch.testing.assert_close(actual, expected_marginals.mean(dim=-1))
    assert forward.call_args.args[0].shape == torch.Size([1, q, 1, 1])

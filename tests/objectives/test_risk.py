"""Tests for robust risk and quality aggregations."""

import pytest
import torch

from robotorchan.objectives import (
    CVaR,
    Expectation,
    MeanVariance,
    SNRatio,
    VaR,
    WorstCase,
    make_risk_measure,
)


def test_basic_risk_measures_reduce_scenario_axis() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    assert torch.allclose(Expectation()(values), torch.tensor([2.0, 4.0]))
    assert torch.allclose(WorstCase()(values), torch.tensor([1.0, 2.0]))
    expected = values.mean(dim=-1) - values.var(dim=-1, unbiased=False)
    assert torch.allclose(MeanVariance()(values), expected)


def test_var_and_cvar_use_lower_tail_for_maximization() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    var = VaR(alpha=0.8)(values)
    cvar = CVaR(alpha=0.8)(values)
    assert torch.allclose(var, torch.tensor([1.8]))
    assert torch.allclose(cvar, torch.tensor([1.0]))


def test_sn_ratio_larger_and_smaller_are_taguchi_forms() -> None:
    values = torch.tensor([[1.0, 2.0]])
    larger_loss = (values.square().reciprocal()).mean(dim=-1)
    smaller_loss = values.square().mean(dim=-1)
    assert torch.allclose(SNRatio("larger_is_better")(values), -10 * torch.log10(larger_loss))
    assert torch.allclose(SNRatio("smaller_is_better")(values), -10 * torch.log10(smaller_loss))


def test_nominal_sn_ratio_requires_target() -> None:
    with pytest.raises(ValueError, match="target is required"):
        SNRatio("nominal_is_best")


def test_nominal_sn_ratio_penalizes_target_deviation() -> None:
    values = torch.tensor([[9.0, 10.0, 11.0]])
    result = SNRatio("nominal_is_best", target=10.0)(values)
    expected_loss = torch.tensor([(1.0 + 0.0 + 1.0) / 3.0])
    assert torch.allclose(result, -10 * torch.log10(expected_loss))


def test_factory_builds_requested_measure() -> None:
    assert isinstance(make_risk_measure("expectation"), Expectation)
    assert isinstance(make_risk_measure("cvar", alpha=0.95), CVaR)
    assert isinstance(make_risk_measure("sn_ratio", sn_type="larger_is_better"), SNRatio)


@pytest.mark.parametrize("measure", [VaR, CVaR])
def test_tail_measures_validate_alpha(measure: type[VaR] | type[CVaR]) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        measure(alpha=1.0)

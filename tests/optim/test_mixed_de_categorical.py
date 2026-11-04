"""Categorical differential evolution regression tests."""

from types import SimpleNamespace

import pytest
import torch

from robotorchan.optim import optimize_acqf
from robotorchan.optim.backends import differential_evolution as de_backend
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


@pytest.mark.parametrize("q", [1, 2])
def test_de_decodes_categorical_indices_before_evaluation(monkeypatch, q):
    bounds = torch.tensor([[0.0, 1.0, 10.0], [1.0, 3.0, 90.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (10.0, 90.0, 30.0)},
    )
    seen = []

    def fake_de(objective, scipy_bounds, **kwargs):
        assert scipy_bounds[2] == (0.0, 2.0)
        encoded = [0.5, 2.4, 1.7] * q
        seen.append(objective(encoded))
        return SimpleNamespace(x=encoded)

    monkeypatch.setattr(de_backend, "differential_evolution", fake_de)
    candidate, value = optimize_acqf(
        SumAcquisition(),
        bounds,
        q,
        optimizer="de",
        variable_space=space,
    )
    assert len(seen) == 1
    assert candidate.shape == (q, 3)
    assert torch.all(candidate[:, 1] == 2.0)
    assert torch.all(candidate[:, 2] == 30.0)
    torch.testing.assert_close(value.reshape(()), candidate.sum())
    assert seen[0] == pytest.approx(-candidate.sum().item())


def test_de_fixed_categorical_feature_uses_public_values(monkeypatch):
    bounds = torch.tensor([[0.0], [90.0]], dtype=torch.double)
    space = MixedVariableSpace(bounds, categorical_values={0: (10.0, 90.0, 30.0)})

    def fake_de(objective, scipy_bounds, **kwargs):
        assert scipy_bounds == [(0.0, 2.0)]
        assert objective([0.0]) == -90.0
        return SimpleNamespace(x=[0.0])

    monkeypatch.setattr(de_backend, "differential_evolution", fake_de)
    candidate, value = optimize_acqf(
        SumAcquisition(),
        bounds,
        1,
        optimizer="de",
        variable_space=space,
        fixed_features={0: 90.0},
    )
    assert candidate.item() == 90.0
    assert value.item() == 90.0

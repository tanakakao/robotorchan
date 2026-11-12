"""Public multi-fidelity GP fit contract regression tests."""

import pytest
import torch

from robotorchan.models import SingleTaskMultiFidelityGP


def test_public_multifidelity_model_exposes_fit() -> None:
    torch.manual_seed(17)
    X = torch.rand(12, 2, dtype=torch.double)
    X[:, -1] = torch.linspace(0.1, 1.0, len(X))
    Y = (X[:, :1] + X[:, 1:2]).sin()
    model = SingleTaskMultiFidelityGP(
        train_X=X,
        train_Y=Y,
        data_fidelities=[-1],
    )
    assert type(model).__module__ == "robotorchan.models.standard.multi_fidelity"
    assert model.supports_fit
    assert callable(model.fit)
    assert callable(model.make_mll)


def test_public_multifidelity_fit_uses_native_mll(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(17)
    X = torch.rand(12, 2, dtype=torch.double)
    X[:, -1] = torch.linspace(0.1, 1.0, len(X))
    Y = (X[:, :1] + X[:, 1:2]).sin()
    model = SingleTaskMultiFidelityGP(X, Y, data_fidelities=[-1])
    seen = []

    def record_fit(mll: object, *, optimizer_kwargs: object) -> None:
        seen.append((mll, optimizer_kwargs))

    monkeypatch.setattr("robotorchan.models.base.fit_gpytorch_mll", record_fit)
    options = {"options": {"maxiter": 2}}
    assert model.fit(optimizer_kwargs=options) is None
    assert len(seen) == 1
    assert seen[0][0].model is model
    assert seen[0][1] == options

from __future__ import annotations

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputPerturbation
from botorch.optim import optimize_acqf

from robotorchan.models import SingleTaskGP
from robotorchan.objectives import make_input_perturbation_objective


def test_single_task_gp_input_perturbation_e2e() -> None:
    dtype = torch.double
    train_X = torch.linspace(0.0, 1.0, 8, dtype=dtype).unsqueeze(-1)
    train_Y = torch.sin(train_X * 6.0)
    perturbation_set = torch.tensor([[-0.03], [0.0], [0.03]], dtype=dtype)
    model = SingleTaskGP(
        train_X, train_Y, input_transform=InputPerturbation(perturbation_set=perturbation_set)
    )
    fit_gpytorch_mll(model.make_mll())

    objective = make_input_perturbation_objective("expectation", n_w=3)
    acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max(), objective=objective)
    X = torch.tensor([[[0.5]]], dtype=dtype, requires_grad=True)
    value = acqf(X)
    assert torch.isfinite(value).all()
    value.sum().backward()
    assert X.grad is not None
    assert torch.isfinite(X.grad).all()

    candidate, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=torch.tensor([[0.1], [0.9]], dtype=dtype),
        q=1,
        num_restarts=2,
        raw_samples=16,
    )
    assert candidate.shape == (1, 1)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(acq_value).all()
    assert torch.all(candidate >= 0.1)
    assert torch.all(candidate <= 0.9)


@pytest.mark.parametrize("risk_type", ["expectation", "worst_case", "var", "cvar"])
def test_input_perturbation_objective_is_botorch_compatible(risk_type: str) -> None:
    objective = make_input_perturbation_objective(risk_type, n_w=3, alpha=0.8)
    samples = torch.randn(8, 2, 3, 1, dtype=torch.double)
    values = objective(samples)
    assert values.shape == (8, 2, 1)
    assert torch.isfinite(values).all()


@pytest.mark.parametrize("risk_type", ["mean_variance", "sn_ratio"])
def test_uncertified_risk_types_fail_explicitly(risk_type: str) -> None:
    with pytest.raises(ValueError, match="not yet certified"):
        make_input_perturbation_objective(risk_type, n_w=3)

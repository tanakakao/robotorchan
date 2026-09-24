from __future__ import annotations

import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputPerturbation
from botorch.optim import optimize_acqf

from robotorchan.models import MultiTaskGP
from robotorchan.objectives import (
    make_input_perturbation_objective,
    make_protected_perturbation_set,
)


def test_multitask_gp_input_perturbation_protects_task_feature() -> None:
    dtype = torch.double
    x = torch.linspace(0.05, 0.95, 6, dtype=dtype)
    train_X = torch.cat(
        [
            torch.stack([x, torch.zeros_like(x)], dim=-1),
            torch.stack([x, torch.ones_like(x)], dim=-1),
        ]
    )
    train_Y = torch.cat(
        [torch.sin(x * 6.0), torch.sin(x * 6.0) + 0.25]
    ).unsqueeze(-1)
    perturbation_set = make_protected_perturbation_set(
        torch.tensor([[-0.03], [0.0], [0.03]], dtype=dtype),
        input_dim=2,
        protected_dims=[1],
    )
    assert torch.count_nonzero(perturbation_set[:, 1]) == 0

    model = MultiTaskGP(
        train_X,
        train_Y,
        task_feature=1,
        output_tasks=[0],
        input_transform=InputPerturbation(perturbation_set=perturbation_set),
    )
    fit_gpytorch_mll(model.make_mll())

    objective = make_input_perturbation_objective("expectation", n_w=3)
    acquisition = qLogExpectedImprovement(model=model, best_f=train_Y[:6].max(), objective=objective)
    candidate, value = optimize_acqf(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1, 0.0], [0.9, 0.0]], dtype=dtype),
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features={1: 0.0},
    )

    assert candidate.shape == (1, 2)
    assert candidate[0, 1].item() == 0.0
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()

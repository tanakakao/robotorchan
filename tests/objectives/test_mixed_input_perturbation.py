from __future__ import annotations

import pytest
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputPerturbation
from botorch.optim import optimize_acqf_mixed

from robotorchan.models import MixedSingleTaskGP
from robotorchan.objectives import (
    make_input_perturbation_objective,
    make_protected_perturbation_set,
)


def test_protected_perturbation_set_preserves_categorical_dimension() -> None:
    perturbations = torch.tensor([[-0.03], [0.0], [0.03]], dtype=torch.double)
    expanded = make_protected_perturbation_set(
        perturbations, input_dim=2, protected_dims=[-1]
    )
    torch.testing.assert_close(expanded[:, 0], perturbations[:, 0])
    assert torch.count_nonzero(expanded[:, 1]) == 0


@pytest.mark.parametrize("protected_dims", [[2], [-3]])
def test_protected_perturbation_set_rejects_out_of_range_dims(
    protected_dims: list[int],
) -> None:
    perturbations = torch.tensor([[0.0]], dtype=torch.double)
    with pytest.raises(ValueError, match="outside the input dimension range"):
        make_protected_perturbation_set(
            perturbations, input_dim=2, protected_dims=protected_dims
        )


def test_mixed_single_task_gp_input_perturbation_e2e() -> None:
    dtype = torch.double
    train_X = torch.tensor(
        [
            [0.05, 0.0],
            [0.20, 1.0],
            [0.35, 0.0],
            [0.50, 1.0],
            [0.65, 0.0],
            [0.80, 1.0],
            [0.95, 0.0],
        ],
        dtype=dtype,
    )
    train_Y = (torch.sin(train_X[:, :1] * 6.0) + 0.2 * train_X[:, 1:2])
    perturbation_set = make_protected_perturbation_set(
        torch.tensor([[-0.03], [0.0], [0.03]], dtype=dtype),
        input_dim=2,
        protected_dims=[1],
    )
    model = MixedSingleTaskGP(
        train_X,
        train_Y,
        cat_dims=[1],
        input_transform=InputPerturbation(perturbation_set=perturbation_set),
    )
    fit_gpytorch_mll(model.make_mll())

    objective = make_input_perturbation_objective("expectation", n_w=3)
    acquisition = qLogExpectedImprovement(
        model=model, best_f=train_Y.max(), objective=objective
    )
    candidate, value = optimize_acqf_mixed(
        acq_function=acquisition,
        bounds=torch.tensor([[0.1, 0.0], [0.9, 1.0]], dtype=dtype),
        q=1,
        num_restarts=2,
        raw_samples=16,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
    )

    assert candidate.shape == (1, 2)
    assert candidate[0, 1].item() in {0.0, 1.0}
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()

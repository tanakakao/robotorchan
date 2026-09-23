"""Contracts for additive MAP-SAAS multi-fidelity regression."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf

from robotorchan.models import AdditiveMapSaasMultiFidelityGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.rand(12, 5, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 3.0) + 0.2 * train_x[:, 1:2] + 0.1 * train_x[:, 4:5]
    return train_x, train_y


def test_additive_map_saas_multifidelity_structure_and_posterior() -> None:
    train_x, train_y = _data()
    model = AdditiveMapSaasMultiFidelityGP(train_x, train_y, data_fidelities=[-1])
    assert model.design_dims == (0, 1, 2, 3)
    assert model.fidelity_dims == (4,)
    assert torch.equal(model.raw_train_X, train_x)
    assert model.make_mll().model is model
    assert len(model.additive_design_covar_module.kernels) == 4
    for component in model.additive_design_covar_module.kernels:
        base_kernel = component.base_kernel
        assert tuple(base_kernel.active_dims.tolist()) == model.design_dims
        assert base_kernel.ard_num_dims == len(model.design_dims)
        assert hasattr(base_kernel, "inv_lengthscale_prior")
    posterior = model.posterior(train_x[:2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()


def test_additive_map_saas_multifidelity_structural_validation() -> None:
    train_x, train_y = _data()
    model = AdditiveMapSaasMultiFidelityGP(
        train_x,
        train_y,
        iteration_fidelity=-2,
        data_fidelities=[-1],
        num_taus=2,
    )
    assert model.iteration_fidelity == 3
    assert model.data_fidelities == (4,)
    assert model.design_dims == (0, 1, 2)
    assert len(model.additive_design_covar_module.kernels) == 2

    try:
        AdditiveMapSaasMultiFidelityGP(
            train_x,
            train_y,
            iteration_fidelity=-1,
            data_fidelities=[4],
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("overlapping fidelity roles must be rejected")


def test_additive_map_saas_multifidelity_supports_native_mf_kg() -> None:
    train_x, train_y = _data()
    model = AdditiveMapSaasMultiFidelityGP(train_x, train_y, data_fidelities=[-1])
    model.eval()
    cost_model = AffineFidelityCostModel(fidelity_weights={4: 1.0}, fixed_cost=0.1)
    cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

    def project(X: torch.Tensor) -> torch.Tensor:
        projected = X.clone()
        projected[..., 4] = 1.0
        return projected

    bounds = torch.stack((torch.zeros(5, dtype=torch.double), torch.ones(5, dtype=torch.double)))
    _, current_value = optimize_acqf(
        acq_function=PosteriorMean(model),
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=8,
        fixed_features={4: 1.0},
    )
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=4,
        current_value=current_value,
        cost_aware_utility=cost_utility,
        project=project,
    )
    X = torch.rand(1, acquisition.get_augmented_q_batch_size(q=1), 5, dtype=torch.double)
    value = acquisition(X)
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()

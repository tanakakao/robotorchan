import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP

from robotorchan.models import SingleTaskGP


def _multifidelity_model() -> tuple[SingleTaskMultiFidelityGP, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.double)
    low = torch.stack((x, torch.full_like(x, 0.5)), dim=-1)
    high = torch.stack((x, torch.ones_like(x)), dim=-1)
    train_X = torch.cat((low, high), dim=0)
    train_Y = (
        torch.sin(train_X[:, :1] * 5.0)
        + 0.2 * (1.0 - train_X[:, 1:])
    )
    model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[1])
    return model, train_X


def test_native_multifidelity_kg_is_compatible() -> None:
    model, train_X = _multifidelity_model()
    cost_model = AffineFidelityCostModel(fidelity_weights={1: 1.0}, fixed_cost=0.1)
    cost_utility = InverseCostWeightedUtility(cost_model=cost_model)
    target_fidelities = {1: 1.0}

    def project(X: torch.Tensor) -> torch.Tensor:
        projected = X.clone()
        projected[..., 1] = target_fidelities[1]
        return projected

    with torch.no_grad():
        current_value = model.posterior(project(train_X)).mean.max()
    acquisition = qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=4,
        current_value=current_value,
        cost_aware_utility=cost_utility,
        project=project,
    )
    X = torch.rand(
        1,
        acquisition.get_augmented_q_batch_size(q=1),
        2,
        dtype=torch.double,
    )
    X[..., 1] = 0.5

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_cost_utility_accepts_robotorchan_cost_surrogate() -> None:
    train_X = torch.tensor(
        [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    train_cost = (0.1 + train_X[:, 1:]).log()
    cost_model = SingleTaskGP(train_X, train_cost)
    utility = InverseCostWeightedUtility(cost_model=cost_model, use_mean=True)
    X = torch.tensor([[[0.3, 0.5]]], dtype=torch.double)
    deltas = torch.ones(1, 1, dtype=torch.double)

    value = utility(X=X, deltas=deltas)

    assert torch.isfinite(value).all()

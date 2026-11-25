"""Native decoupled acquisition constructor and numerical contracts."""

import pytest
import torch
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
    qMultiFidelityHypervolumeKnowledgeGradient,
)
from botorch.models import ModelListGP, SingleTaskGP

from robotorchan.acquisition import create_acquisition, get_acquisition_class, list_acquisitions
from robotorchan.models.partial_observations import split_partial_observations


@pytest.mark.parametrize(
    "cls", [qHypervolumeKnowledgeGradient, qMultiFidelityHypervolumeKnowledgeGradient]
)
def test_native_decoupled_factory_identity(cls):
    assert cls.__name__ in list_acquisitions()
    assert get_acquisition_class(cls.__name__) is cls


@pytest.mark.parametrize("mask", [[True, False], [False, True], [True, True]])
def test_hvkg_partial_observation_value_and_gradient(mask):
    X = torch.linspace(0, 1, 8, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((X.sin(), X.cos()), dim=-1)
    observed = torch.ones_like(Y, dtype=torch.bool)
    observed[0, 1] = False
    observed[-1, 0] = False
    data = split_partial_observations(X, Y, observed)
    model = ModelListGP(*(SingleTaskGP(x, y) for x, y, _ in data))
    model.eval()
    with torch.no_grad():
        model.posterior(X[:1])
    acq = create_acquisition(
        "qHypervolumeKnowledgeGradient",
        model=model,
        ref_point=torch.tensor([-1.0, -1.0], dtype=X.dtype),
        num_fantasies=2,
        num_pareto=2,
        X_evaluation_mask=torch.tensor([mask]),
    )
    augmented_q = acq.get_augmented_q_batch_size(1)
    assert augmented_q == 5
    augmented = torch.linspace(0.15, 0.85, augmented_q, dtype=X.dtype)
    augmented = augmented.reshape(1, augmented_q, 1).requires_grad_()
    value = acq(augmented)
    assert value.shape == (1,)
    assert torch.isfinite(value).all()
    (gradient,) = torch.autograd.grad(value.sum(), augmented)
    assert gradient.shape == augmented.shape
    assert torch.isfinite(gradient).all()
    assert acq.extract_candidates(augmented).shape == (1, 1, 1)

"""Tests for nonstationary binary GP classification."""

import torch
from gpytorch.kernels import ScaleKernel

from robotorchan.models.capabilities import RobustnessType
from robotorchan.models.classification import (
    CLASSIFICATION_MODEL_REGISTRY,
    ClassificationRobustnessType,
    NonstationaryBinarySingleTaskGPClassifier,
)
from robotorchan.models.robust.nonstationary import GibbsKernel


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    return train_x, train_y


def test_nonstationary_classifier_uses_shared_gibbs_kernel() -> None:
    train_x, train_y = _training_data()
    model = NonstationaryBinarySingleTaskGPClassifier(train_x, train_y)

    assert isinstance(model.model.covar_module, ScaleKernel)
    assert isinstance(model.gibbs_kernel, GibbsKernel)


def test_local_lengthscale_is_positive_and_input_dependent() -> None:
    train_x, train_y = _training_data()
    model = NonstationaryBinarySingleTaskGPClassifier(train_x, train_y)
    with torch.no_grad():
        model.gibbs_kernel.lengthscale_slope.fill_(1.0)

    lengthscale = model.local_lengthscale(train_x)

    assert lengthscale.shape == train_x.shape
    assert torch.all(lengthscale > 0)
    assert not torch.allclose(lengthscale[0], lengthscale[-1])


def test_nonstationary_classifier_keeps_standard_bernoulli_prediction_contract() -> None:
    train_x, train_y = _training_data()
    model = NonstationaryBinarySingleTaskGPClassifier(train_x, train_y)

    probabilities = model.predict_proba(train_x[:3])

    assert probabilities.shape == torch.Size([3, 2])
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3, dtype=torch.double))


def test_nonstationary_metadata_describes_latent_process_not_label_noise() -> None:
    train_x, train_y = _training_data()
    model = NonstationaryBinarySingleTaskGPClassifier(train_x, train_y)

    expected = frozenset({ClassificationRobustnessType.NONSTATIONARY})
    assert model.classification_robustness == expected
    assert not model.models_observed_label_process
    assert model.preserves_latent_classification_posterior


def test_nonstationary_classifier_registry_capability() -> None:
    entry = CLASSIFICATION_MODEL_REGISTRY["binary.robust.nonstationary"]

    assert entry.family == "robust"
    assert RobustnessType.NONSTATIONARY in entry.capabilities.robustness

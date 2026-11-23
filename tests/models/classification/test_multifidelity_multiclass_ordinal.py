"""Multi-fidelity multiclass and ordinal classification contracts."""

import pytest
import torch
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from gpytorch.kernels import ProductKernel, ScaleKernel

from robotorchan.acquisition.active_learning.classification import (
    PredictiveEntropy,
    qBALD,
    qPredictiveEntropy,
)
from robotorchan.models.classification import (
    MixedMultiFidelityMulticlassGPClassifier,
    MixedMultiFidelityOrdinalGPClassifier,
    MultiFidelityMulticlassGPClassifier,
    MultiFidelityOrdinalGPClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [
        MultiFidelityMulticlassGPClassifier,
        MultiFidelityOrdinalGPClassifier,
        MixedMultiFidelityMulticlassGPClassifier,
        MixedMultiFidelityOrdinalGPClassifier,
    ],
)
def test_multifidelity_fit_predict_probability_and_candidate(model_type):
    torch.manual_seed(27)
    x = torch.rand(15, 4, dtype=torch.double)
    mixed = model_type in (
        MixedMultiFidelityMulticlassGPClassifier,
        MixedMultiFidelityOrdinalGPClassifier,
    )
    if mixed:
        x[:, 1] = torch.arange(15) % 2
    y = torch.arange(15) % 3
    kwargs = {"cat_dims": [1]} if mixed else {}
    model = model_type(x, y, num_classes=3, iteration_fidelity=-1, **kwargs)
    kernel = (
        model.class_models[0].model.covar_module
        if isinstance(
            model, (MultiFidelityMulticlassGPClassifier, MixedMultiFidelityMulticlassGPClassifier)
        )
        else model.model.covar_module
    )
    assert isinstance(kernel, ScaleKernel)
    assert isinstance(kernel.base_kernel, ProductKernel)
    assert any(isinstance(part, ExponentialDecayKernel) for part in kernel.base_kernel.kernels)
    model.fit(epochs=2, lr=0.005)
    candidates = x[:8].clone()
    with torch.no_grad():
        probabilities = model.predict_proba(candidates)
        entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
        selected = candidates[entropy.argmax()]
    assert probabilities.shape == (8, 3)
    torch.testing.assert_close(
        probabilities.sum(-1),
        torch.ones(8, dtype=torch.double),
        atol=1e-5,
        rtol=1e-5,
    )
    assert selected.shape == (4,)


@pytest.mark.parametrize(
    "model_type",
    [MultiFidelityMulticlassGPClassifier, MultiFidelityOrdinalGPClassifier],
)
def test_data_fidelity_uses_native_downsampling_kernel(model_type):
    x = torch.rand(9, 3, dtype=torch.double)
    y = torch.arange(9) % 3
    model = model_type(x, y, num_classes=3, data_fidelities=[-1])
    kernel = (
        model.class_models[0].model.covar_module
        if model_type is MultiFidelityMulticlassGPClassifier
        else model.model.covar_module
    )
    assert any(isinstance(part, DownsamplingKernel) for part in kernel.base_kernel.kernels)


@pytest.mark.parametrize(
    "model_type",
    [MultiFidelityMulticlassGPClassifier, MultiFidelityOrdinalGPClassifier],
)
def test_fidelity_rejects_missing_overlap_and_unsupported_linear(model_type):
    x = torch.rand(9, 3, dtype=torch.double)
    y = torch.arange(9) % 3
    with pytest.raises(ValueError, match="At least one fidelity"):
        model_type(x, y, num_classes=3)
    with pytest.raises(ValueError, match="distinct"):
        model_type(x, y, num_classes=3, iteration_fidelity=-1, data_fidelities=[-1])
    with pytest.raises(ValueError, match="linear_truncated"):
        model_type(x, y, num_classes=3, iteration_fidelity=-1, linear_truncated=True)


@pytest.mark.parametrize(
    "model_type",
    [MixedMultiFidelityMulticlassGPClassifier, MixedMultiFidelityOrdinalGPClassifier],
)
def test_categorical_fidelity_overlap_rejected(model_type):
    x = torch.rand(9, 3, dtype=torch.double)
    y = torch.arange(9) % 3
    with pytest.raises(ValueError):
        model_type(x, y, num_classes=3, cat_dims=[2], iteration_fidelity=-1)


@pytest.mark.parametrize(
    "model_type",
    [MultiFidelityMulticlassGPClassifier, MultiFidelityOrdinalGPClassifier],
)
def test_multifidelity_classification_acquisition_and_probability_posterior(model_type):
    """Reuse the existing acquisition APIs without new fidelity-specific wrappers."""
    torch.manual_seed(39)
    x = torch.rand(12, 3, dtype=torch.double)
    x[:, -1] = (torch.arange(12) % 2).to(torch.double)
    y = torch.arange(12) % 3
    model = model_type(x, y, num_classes=3, data_fidelities=[-1])
    assert model.data_fidelities == (2,)
    assert model.fidelity_dims == (2,)
    assert model.design_dims == (0, 1)
    model.fit(epochs=2, lr=0.005)
    pool = torch.rand(7, 3, dtype=torch.double)
    pool[:, -1] = 1.0
    single = pool.unsqueeze(-2)
    with torch.no_grad():
        acquisition = PredictiveEntropy(model)
        scores = acquisition(single)
        q_scores = qPredictiveEntropy(model)(pool[:6].reshape(3, 2, 3))
        joint_scores = qBALD(model, num_samples=8)(pool[:6].reshape(3, 2, 3))
        if isinstance(model, MultiFidelityMulticlassGPClassifier):
            probability_posterior = model.probability_posterior(
                pool, num_samples=16, seed=10
            )
        else:
            probability_posterior = model.probability_posterior(pool)
    assert scores.shape == (7,)
    assert q_scores.shape == (3,)
    assert joint_scores.shape == (3,)
    assert probability_posterior.mean.shape == (7, 3)
    assert probability_posterior.variance.shape == (7, 3)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(joint_scores).all()
    assert torch.isfinite(probability_posterior.variance).all()
    assert pool[scores.argmax()].shape == (3,)


@pytest.mark.parametrize(
    "model_type",
    [MixedMultiFidelityMulticlassGPClassifier, MixedMultiFidelityOrdinalGPClassifier],
)
def test_mixed_multifidelity_metadata_and_data_fidelity(model_type):
    x = torch.rand(9, 4, dtype=torch.double)
    x[:, 1] = (torch.arange(9) % 2).to(torch.double)
    y = torch.arange(9) % 3
    model = model_type(
        x, y, num_classes=3, cat_dims=[1], data_fidelities=[-1]
    )
    assert model.cat_dims == (1,)
    assert model.fidelity_dims == (3,)
    assert model.data_fidelities == (3,)
    assert model.design_dims == (0, 1, 2)
    model.fit(epochs=2, lr=0.005)
    with torch.no_grad():
        probabilities = model.predict_proba(x[:4])
    assert probabilities.shape == (4, 3)
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(4, dtype=torch.double),
        atol=1e-5,
        rtol=1e-5,
    )

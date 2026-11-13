"""Focused tests for contextual and hierarchical binary GP classifiers."""

import pytest
import torch
from gpytorch.kernels import AdditiveKernel
from gpytorch.mlls import VariationalELBO

from robotorchan.models.classification import (
    ContextualBinaryGPClassifier,
    HierarchicalBinaryGPClassifier,
    get_classification_model_class,
)


def _training_data():
    x = torch.linspace(0.05, 0.95, 16, dtype=torch.double)
    group = torch.tensor([0, 1] * 8, dtype=torch.double)
    subgroup = torch.tensor([0, 2, 1, 3] * 4, dtype=torch.double)
    X = torch.stack((x, group, subgroup), dim=-1)
    Y = (x + group * 0.15 > 0.5).long()
    return X, Y


@pytest.mark.parametrize(
    ("model_class", "kwargs", "registry_id"),
    [
        (ContextualBinaryGPClassifier, {"context_feature": 1}, "binary.contextual"),
        (
            HierarchicalBinaryGPClassifier,
            {"group_feature": 1, "subgroup_feature": 2},
            "binary.hierarchical",
        ),
    ],
)
def test_contextual_hierarchical_fit_and_registry(model_class, kwargs, registry_id):
    X, Y = _training_data()
    model = model_class(X, Y, inducing_points=8, **kwargs)
    assert isinstance(model.model.covar_module, AdditiveKernel)
    assert isinstance(model.make_mll(), VariationalELBO)
    assert get_classification_model_class(registry_id) is model_class
    model.fit(epochs=2, lr=0.01)
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=torch.double))
    assert torch.isfinite(model.posterior(X[:3]).mean).all()


def test_hierarchical_rejects_subgroup_shared_by_multiple_groups():
    X, Y = _training_data()
    X[:, 2] = 0
    with pytest.raises(ValueError, match="exactly one group"):
        HierarchicalBinaryGPClassifier(X, Y, group_feature=1, subgroup_feature=2)


def test_contextual_rejects_noninteger_context():
    X, Y = _training_data()
    X[0, 1] = 0.5
    with pytest.raises(ValueError, match="integer identifiers"):
        ContextualBinaryGPClassifier(X, Y, context_feature=1)

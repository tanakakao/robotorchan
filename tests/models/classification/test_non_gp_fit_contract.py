"""Fit and probability contracts for optional non-GP binary classifiers."""

import pytest
import torch

pytest.importorskip("sklearn")

from robotorchan.models.classification.binary.non_gp.gradient_boosting import (
    BootstrapGradientBoostingBinaryClassifier,
)
from robotorchan.models.classification.binary.non_gp.sklearn import (
    ExtraTreesBinaryClassifier,
    GradientBoostingBinaryClassifier,
    HistGradientBoostingBinaryClassifier,
    RandomForestBinaryClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [
        ExtraTreesBinaryClassifier,
        GradientBoostingBinaryClassifier,
        HistGradientBoostingBinaryClassifier,
        RandomForestBinaryClassifier,
    ],
)
def test_sklearn_binary_fit_contract(model_type: type) -> None:
    torch.manual_seed(14)
    X = torch.rand(24, 2, dtype=torch.double)
    Y = (X[:, 0] > 0.5).to(dtype=X.dtype)
    model = model_type(X, Y)
    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(X[:2])
    assert model.fit() is None
    assert model.is_fitted
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=X.dtype))


def test_bootstrap_binary_fit_contract() -> None:
    torch.manual_seed(15)
    X = torch.rand(32, 2, dtype=torch.double)
    Y = (X[:, 0] > 0.5).to(dtype=X.dtype)
    model = BootstrapGradientBoostingBinaryClassifier(
        X, Y, n_members=3, random_state=4
    )
    assert not model.is_fitted
    assert model.fit() is None
    assert model.is_fitted
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=X.dtype))

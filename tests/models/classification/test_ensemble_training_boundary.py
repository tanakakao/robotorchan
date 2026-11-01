"""Training boundary contracts for binary classification ensembles."""

import torch

from robotorchan.models.classification.binary.ensemble.gp import (
    GPBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.ensemble.heterogeneous import (
    HeterogeneousBinaryClassificationEnsemble,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


def _members() -> tuple[BinarySingleTaskGPClassifier, BinarySingleTaskGPClassifier]:
    torch.manual_seed(18)
    X = torch.rand(12, 2, dtype=torch.double)
    Y = (X[:, 0] > 0.5).to(dtype=X.dtype)
    first = BinarySingleTaskGPClassifier(X, Y, inducing_points=5)
    second = BinarySingleTaskGPClassifier(X, Y, inducing_points=5)
    first.fit(epochs=2, lr=0.01)
    second.fit(epochs=2, lr=0.01)
    return first, second


def test_gp_ensemble_uses_fitted_member_probabilities() -> None:
    first, second = _members()
    ensemble = GPBinaryClassificationEnsemble(first, second)
    assert not callable(getattr(ensemble, "fit", None))
    X = first.raw_train_X[:3]
    probabilities = ensemble.predict_proba(X)
    expected = (first.predict_proba(X) + second.predict_proba(X)) / 2
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.shape == (3, 2)


def test_heterogeneous_ensemble_preserves_member_training_boundary() -> None:
    first, second = _members()
    ensemble = HeterogeneousBinaryClassificationEnsemble(
        first, second, weights=torch.tensor([1.0, 3.0])
    )
    assert not callable(getattr(ensemble, "fit", None))
    X = first.raw_train_X[:3]
    probabilities = ensemble.predict_proba(X)
    expected = (first.predict_proba(X) + 3 * second.predict_proba(X)) / 4
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.shape == (3, 2)

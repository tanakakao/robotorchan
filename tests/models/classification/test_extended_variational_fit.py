"""Training API contracts for inherited binary classifier variants."""

import pytest
import torch

from robotorchan.models.classification.binary.high_dimensional.map_saas import (
    MapSaasBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.robust.label_noise import (
    LabelNoiseBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.uncertain.uncertain_input import (
    ContinuousUncertainInputBinarySingleTaskGPClassifier,
)


@pytest.mark.parametrize(
    "model_type",
    [
        ContinuousUncertainInputBinarySingleTaskGPClassifier,
        LabelNoiseBinarySingleTaskGPClassifier,
        MapSaasBinarySingleTaskGPClassifier,
    ],
)
def test_inherited_variational_fit(model_type: type) -> None:
    torch.manual_seed(12)
    X = torch.rand(10, 2, dtype=torch.double)
    Y = (X[:, 0] > 0.5).to(dtype=X.dtype)
    model = model_type(X, Y)
    assert model.supports_fit
    assert model.fit(epochs=2, lr=0.01, batch_size=5) is None
    assert not model.training
    assert not model.likelihood.training
    probabilities = model.predict_proba(X[:3])
    assert probabilities.shape == (3, 2)
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3, dtype=X.dtype))

"""Regression tests for multiclass training-objective ownership."""

import pytest
import torch

from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)


@pytest.mark.parametrize("mixed", [False, True])
def test_multiclass_does_not_expose_unrelated_single_gp_mll(mixed: bool) -> None:
    train_x = torch.linspace(0, 1, 9, dtype=torch.double).unsqueeze(-1)
    train_y = torch.arange(9) % 3
    if mixed:
        categories = (torch.arange(9) % 2).to(train_x).unsqueeze(-1)
        train_x = torch.cat((train_x, categories), dim=-1)
        model = MixedMulticlassSingleTaskGPClassifier(
            train_x, train_y, num_classes=3, cat_dims=[1], inducing_points=4
        )
    else:
        model = MulticlassSingleTaskGPClassifier(
            train_x, train_y, num_classes=3, inducing_points=4
        )

    assert model.supports_fit is True
    assert model.supports_mll is False
    with pytest.raises(NotImplementedError, match="class-wise variational objective"):
        model.make_mll()

    # The public fit and posterior routes remain operational.
    model.fit(epochs=1, lr=0.01)
    assert torch.isfinite(model.posterior(train_x[:2]).mean).all()

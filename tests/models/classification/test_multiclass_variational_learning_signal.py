"""Regression tests for learnable multiclass variational GP objectives."""

import pytest
import torch
import torch.nn.functional as F

from robotorchan.models.classification import (
    MixedMulticlassSingleTaskGPClassifier,
    MulticlassSingleTaskGPClassifier,
)


@pytest.mark.parametrize("mixed", [False, True])
def test_multiclass_fit_improves_deterministic_training_log_score(mixed: bool) -> None:
    # Separated, balanced classes provide a known learnable classification task.
    torch.manual_seed(2026)
    centers = torch.tensor([-0.8, 0.0, 0.8], dtype=torch.double)
    X = torch.cat(
        [center + torch.linspace(-0.08, 0.08, 8, dtype=torch.double) for center in centers]
    ).unsqueeze(-1)
    y = torch.arange(3).repeat_interleave(8)
    if mixed:
        categorical = (torch.arange(24) % 2).to(X).unsqueeze(-1)
        X = torch.cat((X, categorical), dim=-1)
        model = MixedMulticlassSingleTaskGPClassifier(
            X, y, num_classes=3, cat_dims=[1], inducing_points=12
        )
    else:
        model = MulticlassSingleTaskGPClassifier(X, y, num_classes=3, inducing_points=12)

    # Evaluate a noise-free diagnostic, separate from stochastic training samples.
    model.eval()
    with torch.no_grad():
        initial_logits = model.posterior(X).mean
        initial_loss = F.cross_entropy(initial_logits, y)

    model.fit(epochs=60, lr=0.03, batch_size=24)
    with torch.no_grad():
        final_logits = model.posterior(X).mean
        final_loss = F.cross_entropy(final_logits, y)
    assert torch.isfinite(final_loss)
    assert final_loss < initial_loss - 0.05
    assert not model.training

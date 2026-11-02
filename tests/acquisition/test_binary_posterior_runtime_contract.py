"""Runtime posterior contract smoke tests for a standard binary GP classifier."""

import torch

from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)


def test_binary_gp_latent_and_probability_contracts() -> None:
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    candidates = torch.tensor([[0.2], [0.8]], dtype=torch.double)

    with torch.no_grad():
        latent = model.posterior(candidates)
        probabilities = model.predict_proba(candidates)
        samples = model.sample_class_probabilities(
            candidates,
            sample_shape=torch.Size([3]),
        )

    assert latent.mean.shape == torch.Size([2, 1])
    assert latent.variance.shape == torch.Size([2, 1])
    assert probabilities.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([3, 2, 2])
    assert torch.isfinite(latent.mean).all()
    assert torch.isfinite(latent.variance).all()
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(samples).all()
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, dtype=torch.double))
    torch.testing.assert_close(samples.sum(dim=-1), torch.ones(3, 2, dtype=torch.double))

"""Small end-to-end RGPE integration test with real GP models."""

import torch
from botorch.models import SingleTaskGP

from robotorchan.optim.transfer.loo import target_loo_predictions
from robotorchan.optim.transfer.rgpe import rgpe_weights
from robotorchan.optim.transfer.rgpe_model import RGPEModel


def test_rgpe_real_gp_end_to_end():
    x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    y = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.double)
    source = SingleTaskGP(x, y)
    target = SingleTaskGP(x, y + 0.1)

    loo = target_loo_predictions(x, y + 0.1)
    assert loo.shape == (3,)
    assert torch.isfinite(loo).all()

    weights = rgpe_weights([source], x, y + 0.1, loo, n_bootstrap=16)
    assert weights.shape == (2,)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(), weights.new_tensor(1.0))

    ensemble = RGPEModel([source], target, weights)
    query = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    posterior = ensemble.posterior(query)
    assert posterior.mean.shape == (2, 1)
    assert posterior.variance.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert (posterior.variance >= 0).all()

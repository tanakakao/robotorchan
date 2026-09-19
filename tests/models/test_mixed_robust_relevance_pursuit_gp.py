import torch
from gpytorch.kernels import AdditiveKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import MixedRobustRelevancePursuitSingleTaskGP


def test_mixed_robust_uses_native_covariance_and_preserves_contract():
    X = torch.tensor([[0.1, 0.0], [0.3, 1.0], [0.6, 0.0], [0.9, 1.0]], dtype=torch.double)
    Y = torch.sin(X[:, :1])
    model = MixedRobustRelevancePursuitSingleTaskGP(X, Y, cat_dims=[-1])
    assert model.cat_dims == (1,)
    assert isinstance(model.covar_module, AdditiveKernel)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    torch.testing.assert_close(model.raw_train_X, X)
    model.eval()
    model.likelihood.eval()
    posterior = model.posterior(X[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()

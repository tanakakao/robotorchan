import pytest
import torch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.ensemble import EnsemblePosterior
from torch import Tensor, nn

from robotorchan.models.non_gp.posterior import make_ensemble_posterior


class DeterministicEnsembleModel(Model, nn.Module):
    """Small differentiable ensemble used to test the BoTorch boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("slopes", torch.tensor([0.5, 1.0, 1.5], dtype=torch.double))

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: Tensor, output_indices=None, observation_noise=False, **kwargs):
        del output_indices, observation_noise, kwargs
        values = self.slopes.reshape(-1, 1, 1) * X[..., :1].unsqueeze(-3)
        return make_ensemble_posterior(values)


def test_make_ensemble_posterior_preserves_shape_dtype_device_and_statistics() -> None:
    values = torch.tensor(
        [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]], dtype=torch.double
    )

    posterior = make_ensemble_posterior(values)

    assert isinstance(posterior, EnsemblePosterior)
    assert posterior.values.shape == torch.Size([3, 2, 1])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert posterior.variance.shape == torch.Size([2, 1])
    assert posterior.dtype == torch.double
    assert posterior.device == values.device
    assert torch.allclose(posterior.mean.squeeze(-1), torch.tensor([3.0, 4.0], dtype=torch.double))


def test_ensemble_posterior_rsample_has_botorch_sample_shape() -> None:
    values = torch.arange(12, dtype=torch.double).reshape(3, 4, 1)
    posterior = make_ensemble_posterior(values)

    samples = posterior.rsample(sample_shape=torch.Size([7]))

    assert samples.shape == torch.Size([7, 4, 1])
    assert samples.dtype == torch.double


def test_ensemble_posterior_keeps_candidate_input_gradients() -> None:
    model = DeterministicEnsembleModel()
    X = torch.tensor([[0.25], [0.75]], dtype=torch.double, requires_grad=True)

    posterior = model.posterior(X)
    posterior.mean.sum().backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    assert torch.count_nonzero(X.grad) == X.numel()


def test_ensemble_model_integrates_with_mc_qei() -> None:
    model = DeterministicEnsembleModel()
    X = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    acqf = qExpectedImprovement(model=model, best_f=0.0)

    value = acqf(X.unsqueeze(0))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_make_ensemble_posterior_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        make_ensemble_posterior(torch.ones(3, 2))
    with pytest.raises(ValueError, match="floating-point"):
        make_ensemble_posterior(torch.ones(3, 2, 1, dtype=torch.long))

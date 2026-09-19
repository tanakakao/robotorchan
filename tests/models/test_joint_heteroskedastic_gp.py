"""Tests for joint variational heteroskedastic GP."""

import torch
from botorch.acquisition import qUpperConfidenceBound

from robotorchan.models import JointHeteroskedasticSingleTaskGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    X = torch.linspace(0.0, 1.0, 12, dtype=torch.double).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X) + (0.02 + 0.1 * X) * torch.randn_like(X)
    return X, Y


def test_joint_training_loss_is_finite_and_differentiable() -> None:
    X, Y = _data()
    model = JointHeteroskedasticSingleTaskGP(X, Y, num_inducing=6, num_mc_samples=4)
    loss = model.training_loss()
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    response_grads = [p.grad for p in model.response_model.parameters() if p.requires_grad]
    noise_grads = [p.grad for p in model.noise_model.parameters() if p.requires_grad]
    assert any(grad is not None for grad in response_grads)
    assert any(grad is not None for grad in noise_grads)


def test_joint_posterior_and_noise_shapes() -> None:
    X, Y = _data()
    model = JointHeteroskedasticSingleTaskGP(X, Y, num_inducing=6)
    test_X = torch.tensor([[[0.2], [0.4]], [[0.6], [0.8]]], dtype=torch.double)
    posterior = model.posterior(test_X)
    noise = model.predicted_noise(test_X)
    assert posterior.mean.shape == torch.Size([2, 2, 1])
    assert noise.shape == torch.Size([2, 2, 1])
    assert torch.isfinite(noise).all()
    assert torch.all(noise >= model.noise_floor)


def test_joint_model_is_accepted_by_mc_acquisition() -> None:
    X, Y = _data()
    model = JointHeteroskedasticSingleTaskGP(X, Y, num_inducing=6)
    candidate = torch.tensor([[[0.5]]], dtype=torch.double)
    value = qUpperConfidenceBound(model=model, beta=0.2)(candidate)
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_joint_state_dict_round_trip() -> None:
    X, Y = _data()
    model = JointHeteroskedasticSingleTaskGP(X, Y, num_inducing=6)
    model.eval()
    expected_mean = model.posterior(X).mean.detach().clone()
    expected_noise = model.predicted_noise(X).detach().clone()
    state = model.state_dict()
    restored = JointHeteroskedasticSingleTaskGP(X, Y, num_inducing=6)
    restored.load_state_dict(state)
    restored.eval()
    assert torch.allclose(expected_mean, restored.posterior(X).mean)
    assert torch.allclose(expected_noise, restored.predicted_noise(X))

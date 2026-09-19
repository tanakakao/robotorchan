"""Tests for the Student-t observation GP."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models.student_t import StudentTSingleTaskGP


def _data(dtype: torch.dtype = torch.double) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.linspace(0, 1, 12, dtype=dtype).unsqueeze(-1)
    Y = torch.sin(2 * torch.pi * X)
    return X, Y


def test_student_t_training_loss_is_finite_and_differentiable() -> None:
    X, Y = _data()
    model = StudentTSingleTaskGP(X, Y, num_inducing=6, df=4.0)
    loss = model.training_loss()
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert model.likelihood.raw_deg_free.grad is not None


def test_student_t_posterior_supports_batch_q_shape() -> None:
    X, Y = _data()
    model = StudentTSingleTaskGP(X, Y, num_inducing=6)
    test_X = X[:6].reshape(2, 3, 1)
    posterior = model.posterior(test_X)
    assert posterior.mean.shape == torch.Size([2, 3, 1])
    assert torch.isfinite(posterior.mean).all()


def test_student_t_model_is_accepted_by_mc_acquisition() -> None:
    X, Y = _data()
    model = StudentTSingleTaskGP(X, Y, num_inducing=6)
    acquisition = qUpperConfidenceBound(model=model, beta=0.2)
    value = acquisition(X[:3].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_student_t_state_dict_round_trip() -> None:
    X, Y = _data()
    model = StudentTSingleTaskGP(X, Y, num_inducing=6)
    model.eval()
    expected = model.posterior(X).mean.detach().clone()
    state = model.state_dict()
    restored = StudentTSingleTaskGP(X, Y, num_inducing=6)
    restored.load_state_dict(state)
    restored.eval()
    assert torch.allclose(expected, restored.posterior(X).mean)
    assert torch.allclose(model.likelihood.deg_free, restored.likelihood.deg_free)


def test_student_t_dtype_migration() -> None:
    X, Y = _data(torch.float32)
    model = StudentTSingleTaskGP(X, Y, num_inducing=6).double()
    assert model.raw_train_X.dtype == torch.double
    assert model.raw_train_Y.dtype == torch.double
    assert model.posterior(X.double()).mean.dtype == torch.double


def test_student_t_validation() -> None:
    X, Y = _data()
    try:
        StudentTSingleTaskGP(X, Y, df=2.0)
    except ValueError:
        pass
    else:
        raise AssertionError("df <= 2 must be rejected.")

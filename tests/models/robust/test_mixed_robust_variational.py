"""Mixed robust variational surrogate tests."""

import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound

from robotorchan.models import MixedContaminatedSingleTaskGP, MixedStudentTSingleTaskGP


def _data():
    x = torch.linspace(0, 1, 10, dtype=torch.double)
    category = torch.tensor([0, 1] * 5, dtype=torch.double)
    X = torch.stack([x, category], dim=-1)
    Y = (torch.sin(2 * torch.pi * x) + 0.2 * category).unsqueeze(-1)
    return X, Y


def test_mixed_student_t_training_and_acquisition() -> None:
    X, Y = _data()
    model = MixedStudentTSingleTaskGP(X, Y, cat_dims=[1], num_inducing=5)
    loss = model.training_loss()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.cat_dims == (1,)
    acq = qUpperConfidenceBound(model=model, beta=0.2)
    value = acq(X[:2].unsqueeze(0))
    assert torch.isfinite(value).all()


def test_mixed_contamination_training_and_acquisition() -> None:
    X, Y = _data()
    model = MixedContaminatedSingleTaskGP(
        X,
        Y,
        cat_dims=[1],
        num_inducing=5,
        num_likelihood_samples=4,
    )
    loss = model.training_loss()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.cat_dims == (1,)
    acq = qUpperConfidenceBound(model=model, beta=0.2)
    value = acq(X[:2].unsqueeze(0))
    assert torch.isfinite(value).all()

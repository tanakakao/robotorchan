import torch
from botorch.models import SingleTaskGP as BoTorchSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import SingleTaskGP


def test_single_task_gp_is_botorch_compatible() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = train_X.sum(dim=-1, keepdim=True)

    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

    assert isinstance(model, BoTorchSingleTaskGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert model.raw_train_Yvar is None
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()
    assert set(model.raw_data) == {"train_X", "train_Y", "train_Yvar"}


def test_single_task_gp_make_mll() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = train_X.square().sum(dim=-1, keepdim=True)

    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    mll = model.make_mll()

    assert isinstance(mll, ExactMarginalLogLikelihood)
    assert mll.model is model
    assert mll.likelihood is model.likelihood


def test_raw_training_data_are_non_persistent_buffers() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = train_X[:, :1]
    train_Yvar = torch.full_like(train_Y, 1e-4)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
    )
    state_dict = model.state_dict()

    assert "_raw_train_X" not in state_dict
    assert "_raw_train_Y" not in state_dict
    assert "_raw_train_Yvar" not in state_dict
    assert torch.equal(model.raw_train_Yvar, train_Yvar)


def test_raw_training_data_are_captured_before_outcome_transform() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = 10.0 + 4.0 * train_X[:, :1]
    original_Y = train_Y.clone()

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=1),
    )

    assert torch.equal(model.raw_train_Y, original_Y)
    assert not torch.equal(model.train_targets.unsqueeze(-1), original_Y)


def test_raw_training_data_follow_model_dtype() -> None:
    train_X = torch.rand(8, 2, dtype=torch.double)
    train_Y = train_X[:, :1]
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)

    model = model.to(dtype=torch.float32)

    assert model.raw_train_X.dtype == torch.float32
    assert model.raw_train_Y.dtype == torch.float32

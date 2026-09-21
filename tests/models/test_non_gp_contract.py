import torch
from torch import nn

from robotorchan.models.base import (
    ModelTrainingMixin,
    NonGPModelMixin,
    UnsupportedModelOperationError,
)


class DummyNonGPModel(NonGPModelMixin, nn.Module):
    supports_input_gradients = True

    def __init__(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self._store_supervised_training_data(train_X, train_Y)
        self.was_fitted = False

    def fit(self) -> None:
        self.was_fitted = True


def test_non_gp_model_preserves_raw_supervised_training_data() -> None:
    train_X = torch.rand(5, 3, dtype=torch.double)
    train_Y = torch.rand(5, 1, dtype=torch.double)
    model = DummyNonGPModel(train_X, train_Y)

    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert model.raw_train_Yvar is None
    assert model.raw_train_X.data_ptr() != train_X.data_ptr()
    assert model.raw_train_Y.data_ptr() != train_Y.data_ptr()


def test_training_capability_defaults_do_not_change_existing_models() -> None:
    assert ModelTrainingMixin.supports_fit is False


def test_non_gp_model_has_explicit_fit_capability_without_mll() -> None:
    model = DummyNonGPModel(torch.rand(4, 2), torch.rand(4, 1))

    assert model.supports_fit is True
    assert model.supports_mll is False
    assert model.supports_input_gradients is True

    model.fit()

    assert model.was_fitted is True

    try:
        model.make_mll()
    except UnsupportedModelOperationError:
        pass
    else:
        raise AssertionError("Non-GP models must not expose a fake marginal likelihood.")


def test_non_gp_raw_data_follows_dtype_and_is_not_serialized() -> None:
    model = DummyNonGPModel(torch.rand(4, 2), torch.rand(4, 1)).double()

    assert model.raw_train_X.dtype == torch.double
    assert model.raw_train_Y.dtype == torch.double
    assert "_raw_train_X" not in model.state_dict()
    assert "_raw_train_Y" not in model.state_dict()

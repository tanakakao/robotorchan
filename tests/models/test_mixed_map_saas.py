import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    MixedAdditiveMapSaasSingleTaskGP,
    MixedEnsembleMapSaasSingleTaskGP,
)


def _data():
    X = torch.tensor([[0.1, 10.0], [0.3, 20.0], [0.6, 10.0], [0.9, 20.0]], dtype=torch.double)
    Y = torch.sin(X[:, :1])
    return X, Y


@pytest.mark.parametrize("model_cls", [MixedAdditiveMapSaasSingleTaskGP, MixedEnsembleMapSaasSingleTaskGP])
def test_mixed_map_saas_raw_api_and_encoding(model_cls):
    X, Y = _data()
    model = model_cls(X, Y, cat_dims=[-1], num_taus=2)
    assert model.cat_dims == (1,)
    assert model.encoded_input_dim == 3
    torch.testing.assert_close(model.raw_train_X, X)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    encoded = model.input_transform(X)
    assert encoded.shape[-1] == 3
    with pytest.raises(ValueError, match="unseen category"):
        model.input_transform(torch.tensor([[0.2, 30.0]], dtype=torch.double))
    state = model.state_dict()
    assert any("category_values" in key for key in state)
    model.to(dtype=torch.float32)
    assert model.category_values[0].dtype == torch.float32

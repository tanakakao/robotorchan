import torch
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


def _make_model() -> tuple[SingleTaskGP, torch.Tensor, torch.Tensor]:
    train_X = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 3.0)
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    return model, train_X, train_Y


def test_conditioning_preserves_constructor_raw_snapshot() -> None:
    model, train_X, train_Y = _make_model()
    new_X = torch.tensor([[0.33]], dtype=torch.double)
    new_Y = torch.sin(new_X * 3.0)

    conditioned = model.condition_on_observations(X=new_X, Y=new_Y)

    assert isinstance(conditioned, SingleTaskGP)
    assert conditioned.raw_data_names == ("train_X", "train_Y", "train_Yvar")
    assert torch.equal(conditioned.raw_train_X, train_X)
    assert torch.equal(conditioned.raw_train_Y, train_Y)
    assert conditioned.raw_train_Yvar is None
    assert conditioned.train_inputs[0].shape[-2] == train_X.shape[-2] + new_X.shape[-2]
    assert model.train_inputs[0].shape[-2] == train_X.shape[-2]


def test_fantasy_preserves_constructor_raw_snapshot() -> None:
    model, train_X, train_Y = _make_model()
    fantasy_X = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([2]), seed=1234)

    model.eval()
    fantasy_model = model.fantasize(X=fantasy_X, sampler=sampler)

    assert isinstance(fantasy_model, SingleTaskGP)
    assert fantasy_model.raw_data_names == ("train_X", "train_Y", "train_Yvar")
    assert torch.equal(fantasy_model.raw_train_X, train_X)
    assert torch.equal(fantasy_model.raw_train_Y, train_Y)
    assert fantasy_model.raw_train_Yvar is None
    assert fantasy_model.train_inputs[0].shape[-2] == train_X.shape[-2] + fantasy_X.shape[-2]

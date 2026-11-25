import pytest
import torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.sampling.list_sampler import ListSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.partial_observations import (
    append_partial_observations,
    split_partial_observations,
)


def test_split_and_append():
    X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double)
    Y = torch.tensor([[1.0, 2.0], [3.0, float("nan")], [float("nan"), 4.0]])
    mask = torch.isfinite(Y)
    data = split_partial_observations(X, Y, mask, torch.ones_like(Y))
    assert torch.equal(data[0][0], X[:2])
    assert torch.equal(data[1][0], X[[0, 2]])
    updated = append_partial_observations(
        data, X[:1], Y[:1], torch.tensor([[True, False]]), torch.ones_like(Y[:1])
    )
    assert updated[0][0].shape == (3, 1)
    assert updated[1][0] is data[1][0]
    assert updated[0][2].shape == (3, 1)


def test_invalid_observation():
    X = torch.zeros(1, 1)
    Y = torch.full((1, 1), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        split_partial_observations(X, Y, torch.ones_like(Y, dtype=torch.bool))


def test_native_masked_fantasy():
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.cat((X.sin(), X.cos()), dim=-1)
    mask = torch.ones_like(Y, dtype=torch.bool)
    mask[0, 1] = False
    mask[-1, 0] = False
    data = split_partial_observations(X, Y, mask)
    model = ModelListGP(*(SingleTaskGP(x, y) for x, y, _ in data))
    model.eval()
    candidate = X[2:3]
    # Initialize prediction caches for every output, including unobserved ones.
    with torch.no_grad():
        model.posterior(candidate)
    sampler = ListSampler(
        SobolQMCNormalSampler(torch.Size([4]), seed=1),
        SobolQMCNormalSampler(torch.Size([4]), seed=2),
    )
    fantasy = model.fantasize(
        X=candidate, sampler=sampler, evaluation_mask=torch.tensor([[True, False]])
    )
    assert fantasy.models[0].train_inputs[0].shape[-2] == 6
    assert fantasy.models[1].train_inputs[0].shape[-2] == 5
    actual = fantasy.models[1].posterior(candidate).mean
    expected = model.models[1].posterior(candidate).mean
    assert actual.shape == torch.Size([4, 1, 1])
    torch.testing.assert_close(actual, expected.expand_as(actual))
    actual_variance = fantasy.models[1].posterior(candidate).variance
    expected_variance = model.models[1].posterior(candidate).variance
    torch.testing.assert_close(actual_variance, expected_variance.expand_as(actual_variance))

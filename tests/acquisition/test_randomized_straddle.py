import torch

from robotorchan.acquisition import RandomizedStraddle
from robotorchan.models import SingleTaskGP


def test_randomized_straddle_is_reproducible_with_generator() -> None:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    model = SingleTaskGP(train_X, train_Y)
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    value_a = RandomizedStraddle(model, target=0.0, generator=generator_a)(X)
    value_b = RandomizedStraddle(model, target=0.0, generator=generator_b)(X)

    assert torch.allclose(value_a, value_b)
    assert torch.isfinite(value_a).all()

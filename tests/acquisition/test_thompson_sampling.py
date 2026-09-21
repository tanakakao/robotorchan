import pytest
import torch

from robotorchan.acquisition import select_thompson_candidates
from robotorchan.models import SingleTaskGP


def _model() -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y)


def test_select_thompson_candidates_returns_members_of_choice_set() -> None:
    model = _model()
    choices = torch.linspace(0.0, 1.0, 16, dtype=torch.double).unsqueeze(-1)

    selected = select_thompson_candidates(model, choices, num_samples=3)

    assert selected.shape == torch.Size([3, 1])
    assert all(torch.any(torch.all(candidate == choices, dim=-1)) for candidate in selected)
    assert torch.unique(selected, dim=0).shape[0] == 3


def test_select_thompson_candidates_validates_request() -> None:
    model = _model()
    choices = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="without replacement"):
        select_thompson_candidates(model, choices, num_samples=3)

    with pytest.raises(ValueError, match="n x d"):
        select_thompson_candidates(model, choices.unsqueeze(0))

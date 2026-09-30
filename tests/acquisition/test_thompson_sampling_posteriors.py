"""Posterior-sampling contracts for Thompson-style candidate generation."""

import pytest
import torch
from botorch.acquisition.objective import GenericMCObjective

pytest.importorskip("sklearn")

from robotorchan.acquisition import select_thompson_candidates
from robotorchan.models import (
    EnsembleMapSaasSingleTaskGP,
    ModelListGP,
    RandomForestSurrogate,
    SingleTaskGP,
)


def _choices() -> torch.Tensor:
    return torch.linspace(0.0, 1.0, 24, dtype=torch.double).unsqueeze(-1)


def _assert_selected_from_choices(
    selected: torch.Tensor,
    choices: torch.Tensor,
    num_samples: int,
) -> None:
    assert selected.shape == torch.Size([num_samples, choices.shape[-1]])
    assert all(torch.any(torch.all(candidate == choices, dim=-1)) for candidate in selected)


def test_gaussian_gp_max_posterior_sampling_selects_discrete_candidates() -> None:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 5.0)
    model = SingleTaskGP(train_x, train_y)
    choices = _choices()

    selected = select_thompson_candidates(model, choices, num_samples=3)

    _assert_selected_from_choices(selected, choices, 3)
    assert torch.unique(selected, dim=0).shape[0] == 3


def test_multi_output_max_posterior_sampling_uses_scalarizing_objective() -> None:
    train_x = torch.linspace(0.0, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    model = ModelListGP(
        SingleTaskGP(train_x, torch.sin(train_x * 5.0)),
        SingleTaskGP(train_x, torch.cos(train_x * 5.0)),
    )
    objective = GenericMCObjective(
        lambda samples, X=None: 0.4 * samples[..., 0] + 0.6 * samples[..., 1]
    )
    choices = _choices()

    selected = select_thompson_candidates(
        model,
        choices,
        num_samples=3,
        objective=objective,
    )

    _assert_selected_from_choices(selected, choices, 3)


def test_tree_ensemble_max_posterior_sampling_selects_discrete_candidates() -> None:
    train_x = torch.linspace(0.0, 1.0, 16, dtype=torch.double).unsqueeze(-1)
    train_y = torch.sin(train_x * 6.0)
    model = RandomForestSurrogate(
        train_x,
        train_y,
        n_estimators=12,
        random_state=0,
    )
    model.fit()
    choices = _choices()

    selected = select_thompson_candidates(model, choices, num_samples=3)

    _assert_selected_from_choices(selected, choices, 3)


def test_map_saas_ensemble_max_posterior_sampling_collapses_ensemble_batch() -> None:
    torch.manual_seed(17)
    train_x = torch.rand(12, 3, dtype=torch.double)
    train_y = torch.sin(train_x[:, :1] * 5.0)
    model = EnsembleMapSaasSingleTaskGP(train_x, train_y, num_taus=2)
    choices = torch.rand(32, 3, dtype=torch.double)

    selected = select_thompson_candidates(model, choices, num_samples=3)

    _assert_selected_from_choices(selected, choices, 3)
    assert torch.unique(selected, dim=0).shape[0] == 3

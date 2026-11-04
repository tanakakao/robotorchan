"""End-to-end regression coverage for existing mixed sampling and GA backends."""

import pytest
import torch

from robotorchan.optim import optimize_acqf
from robotorchan.optim.domains.variable_space import MixedVariableSpace


class SumAcquisition(torch.nn.Module):
    """Return one scalar score per joint candidate batch."""

    def forward(self, candidates):
        return candidates.sum(dim=(-1, -2))


@pytest.fixture
def mixed_space():
    bounds = torch.tensor([[0.0, 1.0, 0.0], [1.0, 4.0, 3.0]], dtype=torch.double)
    space = MixedVariableSpace(
        bounds,
        integer_dims=(1,),
        categorical_values={2: (0.0, 2.0, 3.0)},
    )
    return bounds, space


@pytest.mark.parametrize("optimizer", ["random", "sobol", "ga"])
@pytest.mark.parametrize("q", [1, 2])
def test_mixed_optimizers_return_legal_joint_candidates(optimizer, q, mixed_space):
    bounds, space = mixed_space
    options = (
        {"population_size": 24, "generations": 4}
        if optimizer == "ga"
        else {"num_samples": 64}
    )
    candidate, value = optimize_acqf(
        SumAcquisition(),
        bounds,
        q,
        optimizer=optimizer,
        variable_space=space,
        seed=17,
        optimizer_options=options,
    )
    assert candidate.shape == (q, 3)
    assert value.numel() == 1
    assert torch.all(candidate >= bounds[0])
    assert torch.all(candidate <= bounds[1])
    assert torch.all(candidate[:, 1] == candidate[:, 1].round())
    assert torch.isin(candidate[:, 2], torch.tensor([0.0, 2.0, 3.0])).all()
    torch.testing.assert_close(value.reshape(()), candidate.sum())


@pytest.mark.parametrize("optimizer", ["random", "sobol", "ga"])
def test_mixed_optimizers_are_seed_reproducible(optimizer, mixed_space):
    bounds, space = mixed_space
    options = (
        {"population_size": 24, "generations": 4}
        if optimizer == "ga"
        else {"num_samples": 64}
    )
    kwargs = {
        "optimizer": optimizer,
        "variable_space": space,
        "seed": 29,
        "optimizer_options": options,
    }
    first_candidate, first_value = optimize_acqf(SumAcquisition(), bounds, 2, **kwargs)
    next_candidate, next_value = optimize_acqf(SumAcquisition(), bounds, 2, **kwargs)
    torch.testing.assert_close(first_candidate, next_candidate)
    torch.testing.assert_close(first_value, next_value)


@pytest.mark.parametrize("optimizer", ["random", "sobol", "ga"])
def test_mixed_optimizers_preserve_fixed_discrete_features(optimizer, mixed_space):
    bounds, space = mixed_space
    options = (
        {"population_size": 24, "generations": 4}
        if optimizer == "ga"
        else {"num_samples": 64}
    )
    candidate, _ = optimize_acqf(
        SumAcquisition(),
        bounds,
        2,
        optimizer=optimizer,
        variable_space=space,
        fixed_features={1: 2.0, 2: 3.0},
        seed=7,
        optimizer_options=options,
    )
    assert torch.all(candidate[:, 1] == 2.0)
    assert torch.all(candidate[:, 2] == 3.0)


@pytest.mark.parametrize("optimizer", ["random", "sobol", "ga"])
def test_mixed_optimizers_reject_invalid_fixed_categories(optimizer, mixed_space):
    bounds, space = mixed_space
    with pytest.raises(ValueError, match="categorical values"):
        optimize_acqf(
            SumAcquisition(),
            bounds,
            1,
            optimizer=optimizer,
            variable_space=space,
            fixed_features={2: 1.0},
        )

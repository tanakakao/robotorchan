"""Multi-output Objective compatibility across posterior sample families."""

import torch
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch.sampling.index_sampler import IndexSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.non_gp.posterior import make_ensemble_posterior
from robotorchan.models.standard.multitask import KroneckerMultiTaskGP
from robotorchan.reduction import OutputPCAReducer


def _kronecker_samples(q: int = 3) -> torch.Tensor:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(3.0 * train_X),
            torch.cos(3.0 * train_X),
            0.5 * train_X,
        ),
        dim=-1,
    )
    model = KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, rank=1)
    candidate = torch.linspace(0.2, 0.8, q, dtype=torch.double).unsqueeze(-1)
    return SobolQMCNormalSampler(torch.Size([16]), seed=31)(model.posterior(candidate))


def test_identity_multi_output_objective_selects_requested_kronecker_outputs() -> None:
    samples = _kronecker_samples()
    objective = IdentityMCMultiOutputObjective(outcomes=[2, 0])

    values = objective(samples)

    torch.testing.assert_close(values, samples[..., [2, 0]])
    assert values.shape == torch.Size([16, 3, 2])


def test_weighted_multi_output_objective_preserves_vector_outputs() -> None:
    samples = _kronecker_samples()
    weights = torch.tensor([2.0, -0.5], dtype=torch.double)
    objective = WeightedMCMultiOutputObjective(weights=weights, outcomes=[0, 2])

    values = objective(samples)

    torch.testing.assert_close(values, samples[..., [0, 2]] * weights)
    assert values.shape == torch.Size([16, 3, 2])


def test_linear_mc_objective_scalarizes_all_selected_information() -> None:
    samples = _kronecker_samples()
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.double)
    objective = LinearMCObjective(weights=weights)

    values = objective(samples)

    torch.testing.assert_close(values, (samples * weights).sum(dim=-1))
    assert values.shape == torch.Size([16, 3])


def test_generic_mc_objective_supports_nonlinear_multi_output_scalarization() -> None:
    samples = _kronecker_samples()
    objective = GenericMCObjective(
        lambda Y, X=None: Y[..., 0].square() + Y[..., 1] * Y[..., 2]
    )

    values = objective(samples)

    expected = samples[..., 0].square() + samples[..., 1] * samples[..., 2]
    torch.testing.assert_close(values, expected)
    assert values.shape == torch.Size([16, 3])


def test_ensemble_multi_output_objective_uses_same_native_api() -> None:
    members = torch.arange(7 * 3 * 3, dtype=torch.double).reshape(7, 3, 3)
    posterior = make_ensemble_posterior(members)
    samples = IndexSampler(torch.Size([16]), seed=37)(posterior)
    objective = IdentityMCMultiOutputObjective(outcomes=[2, 0])

    values = objective(samples)

    torch.testing.assert_close(values, samples[..., [2, 0]])
    assert values.shape == torch.Size([16, 3, 2])


def test_output_reduction_selects_outputs_after_original_space_restoration() -> None:
    train_Y = torch.randn(24, 5, dtype=torch.double)
    reducer = OutputPCAReducer(n_components=2).fit(train_Y)
    latent_samples = torch.randn(16, 3, 2, dtype=torch.double)
    restored = reducer.inverse_transform(latent_samples)
    objective = IdentityMCMultiOutputObjective(outcomes=[4, 1])

    values = objective(restored)

    torch.testing.assert_close(values, restored[..., [4, 1]])
    assert values.shape == torch.Size([16, 3, 2])

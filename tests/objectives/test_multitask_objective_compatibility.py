"""Objective semantics for long-format and block-design multi-task models."""

import torch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models.standard.multitask import KroneckerMultiTaskGP, MultiTaskGP


def _long_format_model() -> MultiTaskGP:
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    task_0 = torch.zeros_like(x)
    task_1 = torch.ones_like(x)
    train_X = torch.cat(
        (
            torch.cat((x, task_0), dim=-1),
            torch.cat((x, task_1), dim=-1),
        ),
        dim=0,
    )
    train_Y = torch.cat((torch.sin(3.0 * x), torch.cos(3.0 * x)), dim=0)
    return MultiTaskGP(train_X=train_X, train_Y=train_Y, task_feature=-1, rank=1)


def _kronecker_model() -> KroneckerMultiTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        (
            torch.sin(3.0 * train_X),
            torch.cos(3.0 * train_X),
        ),
        dim=-1,
    )
    return KroneckerMultiTaskGP(train_X=train_X, train_Y=train_Y, rank=1)


def test_long_format_explicit_task_feature_is_single_output_posterior() -> None:
    model = _long_format_model()
    candidate = torch.tensor([[0.2, 0.0], [0.8, 1.0]], dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=41)(posterior)
    objective = GenericMCObjective(lambda Y, X=None: Y.squeeze(-1))

    values = objective(samples)

    assert posterior.mean.shape == torch.Size([2, 1])
    assert samples.shape == torch.Size([16, 2, 1])
    assert values.shape == torch.Size([16, 2])


def test_long_format_output_indices_create_task_output_axis() -> None:
    model = _long_format_model()
    candidate = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    posterior = model.posterior(candidate, output_indices=[1, 0])
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=43)(posterior)
    objective = IdentityMCMultiOutputObjective(outcomes=[0, 1])

    values = objective(samples)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([16, 2, 2])
    torch.testing.assert_close(values, samples)


def test_long_format_task_feature_is_not_an_objective_output() -> None:
    model = _long_format_model()
    candidate = torch.tensor([[0.3, 1.0]], dtype=torch.double)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=47)(model.posterior(candidate))
    objective = LinearMCObjective(weights=torch.tensor([1.0], dtype=torch.double))

    values = objective(samples)

    assert samples.shape[-1] == 1
    assert values.shape == torch.Size([16, 1])


def test_kronecker_tasks_are_native_posterior_outputs() -> None:
    model = _kronecker_model()
    candidate = torch.tensor([[0.2], [0.8]], dtype=torch.double)
    posterior = model.posterior(candidate)
    samples = SobolQMCNormalSampler(torch.Size([16]), seed=53)(posterior)
    objective = IdentityMCMultiOutputObjective(outcomes=[1, 0])

    values = objective(samples)

    assert posterior.mean.shape == torch.Size([2, 2])
    assert samples.shape == torch.Size([16, 2, 2])
    torch.testing.assert_close(values, samples[..., [1, 0]])


def test_both_multitask_representations_support_same_scalar_objective_contract() -> None:
    long_model = _long_format_model()
    kron_model = _kronecker_model()
    candidate = torch.tensor([[0.25], [0.75]], dtype=torch.double)
    sampler = SobolQMCNormalSampler(torch.Size([16]), seed=59)
    objective = LinearMCObjective(weights=torch.tensor([0.4, 0.6], dtype=torch.double))

    long_values = objective(sampler(long_model.posterior(candidate, output_indices=[0, 1])))
    kron_values = objective(sampler(kron_model.posterior(candidate)))

    assert long_values.shape == torch.Size([16, 2])
    assert kron_values.shape == torch.Size([16, 2])

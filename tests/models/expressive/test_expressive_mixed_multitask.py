"""Runtime contracts for expressive Mixed x MultiTask models."""

import torch

from robotorchan.models import (
    MixedInfiniteWidthBNNMultiTaskGP,
    MixedMultiTaskDeepGP,
    MixedSpectralMixtureMultiTaskGP,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(23)
    continuous = torch.rand(18, 1, dtype=torch.double)
    category = torch.arange(18, dtype=torch.double).remainder(2).unsqueeze(-1)
    task = torch.arange(18, dtype=torch.double).remainder(3).unsqueeze(-1)
    train_X = torch.cat((continuous, category, task), dim=-1)
    train_Y = torch.sin(4.0 * continuous) + 0.2 * category + 0.3 * task
    return train_X, train_Y


def test_mixed_multitask_deep_gp_posterior_and_embedding_gradients() -> None:
    train_X, train_Y = _data()
    model = MixedMultiTaskDeepGP(
        train_X,
        train_Y,
        task_feature=2,
        cat_dims=(1,),
        hidden_dims=(4,),
        num_inducing=6,
        posterior_samples=8,
    )
    posterior = model.posterior(train_X[:4], num_samples=8)
    assert posterior.mean.shape == torch.Size([4, 1])
    assert torch.isfinite(posterior.mean).all()
    loss = model.training_loss(num_likelihood_samples=3)
    loss.backward()
    assert model.task_embedding.weight.grad is not None
    assert model.category_embeddings[0].weight.grad is not None


def test_mixed_multitask_infinite_width_bnn_posterior() -> None:
    train_X, train_Y = _data()
    model = MixedInfiniteWidthBNNMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        cat_dims=[1],
    )
    posterior = model.posterior(train_X[:4])
    assert model.cat_dims == (1,)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_mixed_multitask_spectral_mixture_posterior() -> None:
    train_X, train_Y = _data()
    model = MixedSpectralMixtureMultiTaskGP(
        train_X,
        train_Y,
        task_feature=2,
        cat_dims=[1],
        num_mixtures=2,
    )
    posterior = model.posterior(train_X[:4])
    assert model.cat_dims == (1,)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()

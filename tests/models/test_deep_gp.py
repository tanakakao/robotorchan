"""Tests for the single-task DeepGP foundation."""

import torch
from gpytorch.mlls import DeepApproximateMLL

from robotorchan.models.deep_gp import MultiTaskDeepGP, SingleTaskDeepGP


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    train_X = torch.rand(12, 3, dtype=torch.double)
    train_Y = (
        torch.sin(4.0 * train_X[:, :1]) + 0.4 * train_X[:, 1:2].square() - 0.2 * train_X[:, 2:3]
    )
    return train_X, train_Y


def test_deep_gp_retains_raw_training_data_and_structure() -> None:
    train_X, train_Y = _data()
    model = SingleTaskDeepGP(
        train_X,
        train_Y,
        hidden_dims=(5, 3),
        num_inducing=6,
        random_state=3,
    )

    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
    assert model.raw_train_Yvar is None
    assert model.hidden_dims == (5, 3)
    assert len(model.hidden_layers) == 2
    assert model.hidden_layers[0].output_dims == 5
    assert model.hidden_layers[1].output_dims == 3
    assert model.output_layer.output_dims is None


def test_deep_gp_forward_samples_have_expected_shape() -> None:
    train_X, train_Y = _data()
    model = SingleTaskDeepGP(train_X, train_Y, hidden_dims=(4,), num_inducing=5)

    with torch.no_grad():
        output = model(train_X[:4])

    assert output.mean.shape[-1] == 4
    assert torch.isfinite(output.mean).all()
    assert torch.isfinite(output.variance).all()


def test_deep_gp_make_mll_and_training_loss_are_finite() -> None:
    train_X, train_Y = _data()
    model = SingleTaskDeepGP(train_X, train_Y, hidden_dims=(4,), num_inducing=5)

    assert isinstance(model.make_mll(), DeepApproximateMLL)
    loss = model.training_loss(num_likelihood_samples=3)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_deep_gp_training_loss_backpropagates_through_all_layers() -> None:
    train_X, train_Y = _data()
    model = SingleTaskDeepGP(train_X, train_Y, hidden_dims=(4,), num_inducing=5)

    loss = model.training_loss(num_likelihood_samples=3)
    loss.backward()

    hidden_grad = model.hidden_layers[0].covar_module.base_kernel.raw_lengthscale.grad
    output_grad = model.output_layer.covar_module.base_kernel.raw_lengthscale.grad
    assert hidden_grad is not None
    assert output_grad is not None
    assert torch.isfinite(hidden_grad).all()
    assert torch.isfinite(output_grad).all()


def test_deep_gp_optimizer_step_updates_parameters() -> None:
    train_X, train_Y = _data()
    model = SingleTaskDeepGP(train_X, train_Y, hidden_dims=(4,), num_inducing=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = model.output_layer.covar_module.base_kernel.raw_lengthscale.detach().clone()

    optimizer.zero_grad()
    model.training_loss(num_likelihood_samples=3).backward()
    optimizer.step()

    after = model.output_layer.covar_module.base_kernel.raw_lengthscale.detach()
    assert not torch.equal(before, after)


def test_deep_gp_minibatch_uses_explicit_total_num_data() -> None:
    train_X, train_Y = _data()
    model = SingleTaskDeepGP(train_X, train_Y, hidden_dims=(4,), num_inducing=5)

    loss = model.training_loss(
        train_X[:6],
        train_Y[:6],
        num_data=train_X.shape[-2],
        num_likelihood_samples=3,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_deep_gp_rejects_invalid_configuration() -> None:
    train_X, train_Y = _data()

    for hidden_dims in [(), (4, 0)]:
        try:
            SingleTaskDeepGP(train_X, train_Y, hidden_dims=hidden_dims)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid hidden_dims must raise ValueError")

    try:
        SingleTaskDeepGP(train_X, train_Y, num_inducing=0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive num_inducing must raise ValueError")


def _multitask_data() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    data_X = torch.rand(12, 2, dtype=torch.double)
    tasks = torch.arange(12, dtype=torch.double).remainder(3).unsqueeze(-1)
    train_X = torch.cat((data_X[:, :1], tasks, data_X[:, 1:]), dim=-1)
    train_Y = torch.sin(3.0 * data_X[:, :1]) + 0.2 * data_X[:, 1:] + 0.3 * tasks
    return train_X, train_Y


def test_multitask_deep_gp_preserves_original_training_data_and_task_structure() -> None:
    train_X, train_Y = _multitask_data()
    model = MultiTaskDeepGP(
        train_X,
        train_Y,
        task_feature=1,
        task_embedding_dim=2,
        hidden_dims=(4,),
        num_inducing=5,
    )

    torch.testing.assert_close(model.raw_train_X, train_X)
    torch.testing.assert_close(model.raw_train_Y, train_Y)
    assert model.task_feature == 1
    assert model.num_tasks == 3
    assert model.task_embedding.num_embeddings == 3
    assert model.task_embedding.embedding_dim == 2


def test_multitask_deep_gp_posterior_accepts_original_long_format_inputs() -> None:
    train_X, train_Y = _multitask_data()
    model = MultiTaskDeepGP(
        train_X,
        train_Y,
        task_feature=1,
        hidden_dims=(4,),
        num_inducing=5,
        posterior_samples=8,
    )

    posterior = model.posterior(train_X[:4], num_samples=8)

    assert posterior.mean.shape == torch.Size([4, 1])
    assert posterior.variance.shape == torch.Size([4, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_multitask_deep_gp_training_loss_updates_task_embedding() -> None:
    train_X, train_Y = _multitask_data()
    model = MultiTaskDeepGP(
        train_X,
        train_Y,
        task_feature=1,
        hidden_dims=(4,),
        num_inducing=5,
    )

    loss = model.training_loss(num_likelihood_samples=3)
    loss.backward()

    gradient = model.task_embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_multitask_deep_gp_rejects_invalid_task_encoding() -> None:
    train_X, train_Y = _multitask_data()
    invalid_X = train_X.clone()
    invalid_X[0, 1] = 4.0

    try:
        MultiTaskDeepGP(invalid_X, train_Y, task_feature=1)
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous task indices must raise ValueError")

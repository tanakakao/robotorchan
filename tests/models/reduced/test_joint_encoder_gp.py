import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import JointEncoderGP


def _data():
    torch.manual_seed(801)
    X = torch.rand(20, 7, dtype=torch.double)
    Y = 1.3 * X[:, :1] - X[:, 1:2] + torch.sin(torch.pi * X[:, 2:3])
    return X, Y


def _make_model(X, Y, *, random_state=13):
    return JointEncoderGP(
        X,
        Y,
        latent_dim=3,
        hidden_dims=(8,),
        random_state=random_state,
    )


def test_joint_encoder_gp_keeps_original_training_inputs_and_raw_data():
    X, Y = _data()
    model = _make_model(X, Y)

    torch.testing.assert_close(model.raw_train_X, X)
    torch.testing.assert_close(model.raw_train_Y, Y)
    torch.testing.assert_close(model.train_inputs[0], X)
    assert model.encode(X).shape == (20, 3)
    assert model.make_mll().model is model


def test_joint_encoder_gp_mll_backpropagates_to_encoder():
    X, Y = _data()
    model = _make_model(X, Y)

    loss = model.training_loss()
    assert loss.ndim == 0
    loss.backward()

    encoder_gradients = [parameter.grad for parameter in model.encoder.parameters()]
    assert all(gradient is not None for gradient in encoder_gradients)
    gradients = [gradient for gradient in encoder_gradients if gradient is not None]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_joint_encoder_gp_posterior_and_acquisition_keep_original_x_gradients():
    X, Y = _data()
    model = _make_model(X, Y)
    model.eval()

    candidate = X[:2].clone().requires_grad_(True)
    posterior = model.posterior(candidate)
    assert posterior.mean.shape == (2, 1)

    acquisition = qLogExpectedImprovement(
        model=model,
        best_f=Y.max(),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    )
    value = acquisition(candidate.unsqueeze(0))
    value.sum().backward()

    assert candidate.grad is not None
    assert torch.isfinite(candidate.grad).all()


def test_joint_encoder_gp_state_dict_round_trip():
    X, Y = _data()
    source = _make_model(X, Y, random_state=13)
    target = _make_model(X, Y, random_state=99)

    target.load_state_dict(source.state_dict())

    torch.testing.assert_close(target.encode(X), source.encode(X))
    source.eval()
    target.eval()
    torch.testing.assert_close(target.posterior(X[:4]).mean, source.posterior(X[:4]).mean)


def test_joint_encoder_gp_supports_expansive_latent_dimension():
    X, Y = _data()
    model = JointEncoderGP(
        X,
        Y,
        latent_dim=9,
        hidden_dims=(12,),
        random_state=13,
    )

    assert model.encode(X).shape == (20, 9)
    assert torch.isfinite(model.training_loss())


def test_joint_encoder_gp_accepts_custom_feature_extractor():
    X, Y = _data()
    feature_extractor = torch.nn.Sequential(
        torch.nn.Linear(7, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 4),
    )
    model = JointEncoderGP(
        X,
        Y,
        latent_dim=4,
        feature_extractor=feature_extractor,
        random_state=13,
    )

    assert model.encoder is feature_extractor
    assert model.encode(X).shape == (20, 4)

    loss = model.training_loss()
    loss.backward()
    assert all(parameter.grad is not None for parameter in feature_extractor.parameters())


def test_joint_encoder_gp_rejects_custom_feature_dimension_mismatch():
    X, Y = _data()
    feature_extractor = torch.nn.Linear(7, 5)

    try:
        JointEncoderGP(
            X,
            Y,
            latent_dim=4,
            feature_extractor=feature_extractor,
        )
    except ValueError as error:
        assert "output dimension must equal latent_dim" in str(error)
    else:
        raise AssertionError("Expected a feature-extractor dimension validation error.")


def test_joint_encoder_gp_mc_acquisitions_support_original_space():
    X, Y = _data()
    model = _make_model(X, Y)
    model.eval()
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([16]))
    candidates = X[:2].unsqueeze(0)

    acquisitions = [
        qLogExpectedImprovement(model=model, best_f=Y.max(), sampler=sampler),
        qLogNoisyExpectedImprovement(model=model, X_baseline=X, sampler=sampler),
        qUpperConfidenceBound(model=model, beta=0.2, sampler=sampler),
    ]

    for acquisition in acquisitions:
        value = acquisition(candidates)
        assert value.shape == torch.Size([1])
        assert torch.isfinite(value).all()


def test_joint_encoder_gp_optimize_acqf_runs_in_original_space():
    X, Y = _data()
    model = _make_model(X, Y)
    model.eval()
    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([8])),
    )
    bounds = torch.stack((torch.zeros(7, dtype=X.dtype), torch.ones(7, dtype=X.dtype)))

    candidate, value = optimize_acqf(
        acquisition,
        bounds=bounds,
        q=1,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidate.shape == (1, 7)
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(value).all()


def test_joint_encoder_gp_condition_on_observations_keeps_original_space():
    X, Y = _data()
    model = _make_model(X, Y)
    model.eval()
    X_new = torch.rand(2, 7, dtype=X.dtype)
    Y_new = X_new[:, :1] - 0.5 * X_new[:, 1:2]

    conditioned = model.condition_on_observations(X=X_new, Y=Y_new)
    posterior = conditioned.posterior(X_new)

    assert conditioned.train_inputs[0].shape[-1] == 7
    assert posterior.mean.shape == (2, 1)
    assert torch.isfinite(posterior.mean).all()


def test_joint_encoder_gp_fantasize_keeps_original_space():
    X, Y = _data()
    model = _make_model(X, Y)
    model.eval()
    X_new = torch.rand(2, 7, dtype=X.dtype)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]))

    fantasy = model.fantasize(X_new, sampler=sampler)
    posterior = fantasy.posterior(X_new)

    assert fantasy.train_inputs[0].shape[-1] == 7
    assert posterior.mean.shape[-2:] == (2, 1)
    assert torch.isfinite(posterior.mean).all()

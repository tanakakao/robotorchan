"""Runtime contracts for PCA multi-fidelity regression."""

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models.cost import AffineFidelityCostModel
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler, SobolQMCNormalSampler

from robotorchan.models import (
    MapSaasMultiFidelityGP,
    PCAMultiFidelityGP,
    PLSMultiFidelityGP,
    RandomProjectionMultiFidelityGP,
)


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.rand(10, 5, dtype=torch.double)
    fidelity = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, fidelity), dim=-1)
    train_y = design[:, :2].sum(dim=-1, keepdim=True) + 0.2 * fidelity
    return train_x, train_y


def test_pca_multifidelity_preserves_structural_fidelity() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.fidelity_dims == (5,)
    assert model.design_dims == (0, 1, 2, 3, 4)
    assert model.encoded_fidelity_dims == (2,)
    encoded = model._encode_inputs(train_x)
    assert encoded.shape[-1] == 3
    assert torch.equal(encoded[..., -1], train_x[..., -1])


def test_pca_multifidelity_posterior_sampling_and_mc_acquisition() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )
    model.eval()

    candidates = train_x[:2]
    posterior = model.posterior(candidates)
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()

    acquisition = qUpperConfidenceBound(
        model=model,
        beta=0.2,
        sampler=IIDNormalSampler(sample_shape=torch.Size([8])),
    )
    value = acquisition(candidates[:1].unsqueeze(0))
    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_pca_multifidelity_make_mll() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )

    mll = model.make_mll()
    assert mll.model is model


def test_pca_multifidelity_preserves_iteration_and_data_roles() -> None:
    torch.manual_seed(0)
    design = torch.rand(10, 4, dtype=torch.double)
    iteration = torch.linspace(0.1, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    data = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, iteration, data), dim=-1)
    train_y = design[:, :2].sum(dim=-1, keepdim=True) + 0.1 * iteration + 0.2 * data

    model = PCAMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        iteration_fidelity=-2,
        data_fidelities=[-1],
    )

    assert model.iteration_fidelity == 4
    assert model.data_fidelities == (5,)
    assert model.fidelity_dims == (4, 5)
    assert model.encoded_fidelity_dims == (2, 3)
    encoded = model._encode_inputs(train_x)
    assert torch.equal(encoded[..., 2], train_x[..., 4])
    assert torch.equal(encoded[..., 3], train_x[..., 5])


def test_pca_multifidelity_rejects_overlapping_fidelity_roles() -> None:
    train_x, train_y = _data()
    try:
        PCAMultiFidelityGP(
            train_x,
            train_y,
            n_components=2,
            iteration_fidelity=-1,
            data_fidelities=[-1],
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("Expected overlapping fidelity dimensions to be rejected.")


def test_pls_multifidelity_preserves_structural_fidelity() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = PLSMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
    )

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.fidelity_dims == (5,)
    assert model.design_dims == (0, 1, 2, 3, 4)
    encoded = model._encode_inputs(train_x)
    assert encoded.shape[-1] == 3
    assert torch.equal(encoded[..., -1], train_x[..., -1])

    model.eval()
    posterior = model.posterior(train_x[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()
    assert model.make_mll().model is model


def test_random_projection_multifidelity_preserves_structural_fidelity() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = RandomProjectionMultiFidelityGP(
        train_x,
        train_y,
        n_components=2,
        data_fidelities=[-1],
        random_state=7,
    )

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.fidelity_dims == (5,)
    assert model.design_dims == (0, 1, 2, 3, 4)
    encoded = model._encode_inputs(train_x)
    assert encoded.shape[-1] == 3
    assert torch.equal(encoded[..., -1], train_x[..., -1])

    repeated = model._encode_inputs(train_x)
    assert torch.equal(encoded, repeated)
    model.eval()
    posterior = model.posterior(train_x[:2])
    assert posterior.mean.shape == torch.Size([2, 1])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()
    assert model.make_mll().model is model


def test_map_saas_multifidelity_keeps_saas_on_design_kernel() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model = MapSaasMultiFidelityGP(train_x, train_y, data_fidelities=[-1])

    assert torch.equal(model.raw_train_X, train_x)
    assert torch.equal(model.raw_train_Y, train_y)
    assert model.design_dims == (0, 1, 2, 3, 4)
    assert model.fidelity_dims == (5,)
    assert model.data_fidelities == (5,)
    assert model.saas_design_kernel.ard_num_dims == 5
    assert tuple(model.saas_design_kernel.active_dims.tolist()) == model.design_dims
    prior_names = {name for name, *_ in model.saas_design_kernel.named_priors()}
    assert "tau_prior" in prior_names
    assert model.make_mll().model is model

    model.eval()
    posterior = model.posterior(train_x[:2])
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()
    assert torch.isfinite(posterior.rsample(torch.Size([3]))).all()


def test_map_saas_multifidelity_preserves_fidelity_roles_and_rejects_overlap() -> None:
    torch.manual_seed(0)
    design = torch.rand(10, 4, dtype=torch.double)
    iteration = torch.linspace(0.1, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    data = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, iteration, data), dim=-1)
    train_y = design[:, :2].sum(dim=-1, keepdim=True) + iteration + data

    model = MapSaasMultiFidelityGP(
        train_x,
        train_y,
        iteration_fidelity=-2,
        data_fidelities=[-1],
    )
    assert model.iteration_fidelity == 4
    assert model.data_fidelities == (5,)
    assert model.design_dims == (0, 1, 2, 3)
    assert model.saas_design_kernel.ard_num_dims == 4

    try:
        MapSaasMultiFidelityGP(
            train_x,
            train_y,
            iteration_fidelity=-1,
            data_fidelities=[-1],
        )
    except ValueError as error:
        assert "duplicates" in str(error)
    else:
        raise AssertionError("Expected overlapping fidelity dimensions to be rejected.")


def test_map_saas_multifidelity_rejects_linear_truncated_path() -> None:
    train_x, train_y = _data()
    try:
        MapSaasMultiFidelityGP(
            train_x,
            train_y,
            data_fidelities=[-1],
            linear_truncated=True,
        )
    except ValueError as error:
        assert "linear_truncated=False" in str(error)
    else:
        raise AssertionError("Expected linear-truncated covariance to be rejected.")


def test_reduced_multifidelity_conditioning_accepts_raw_inputs() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        model = model_class(
            train_x,
            train_y,
            n_components=2,
            data_fidelities=[-1],
            **kwargs,
        )
        model.eval()
        new_x = train_x[:2].clone()
        new_y = train_y[:2].clone()
        _ = model.posterior(new_x)
        conditioned = model.condition_on_observations(X=new_x, Y=new_y)

        assert conditioned.train_inputs[0].shape[-1] == 3
        posterior = super(model_class, conditioned).posterior(model._encode_inputs(new_x))
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()


def test_reduced_multifidelity_fantasize_accepts_raw_inputs() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        model = model_class(
            train_x,
            train_y,
            n_components=2,
            data_fidelities=[-1],
            **kwargs,
        )
        model.eval()
        model.likelihood.eval()
        candidates = train_x[:2].clone()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([3]), seed=11)
        fantasy = model.fantasize(X=candidates, sampler=sampler)

        assert fantasy.train_inputs[0].shape[-1] == 3
        assert type(fantasy.input_reducer) is type(model.input_reducer)
        assert fantasy.input_reducer.state_dict().keys() == model.input_reducer.state_dict().keys()
        for key, value in model.input_reducer.state_dict().items():
            torch.testing.assert_close(fantasy.input_reducer.state_dict()[key], value)
        posterior = super(model_class, fantasy).posterior(model._encode_inputs(candidates))
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()


def test_reduced_multifidelity_models_support_native_mf_kg() -> None:
    torch.manual_seed(0)
    train_x, train_y = _data()
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        model = model_class(
            train_x,
            train_y,
            n_components=2,
            data_fidelities=[-1],
            **kwargs,
        )
        model.eval()

        cost_model = AffineFidelityCostModel(fidelity_weights={5: 1.0}, fixed_cost=0.1)
        cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

        def project(X: torch.Tensor) -> torch.Tensor:
            projected = X.clone()
            projected[..., 5] = 1.0
            return projected

        bounds = torch.stack((train_x.min(dim=0).values, train_x.max(dim=0).values))
        bounds[:, 5] = 1.0
        _, current_value = optimize_acqf(
            acq_function=PosteriorMean(model),
            bounds=bounds,
            q=1,
            num_restarts=2,
            raw_samples=8,
            fixed_features={5: 1.0},
        )
        acquisition = qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=4,
            current_value=current_value,
            cost_aware_utility=cost_utility,
            project=project,
        )
        X = torch.rand(
            1,
            acquisition.get_augmented_q_batch_size(q=1),
            train_x.shape[-1],
            dtype=torch.double,
        )
        X[..., 5] = 0.2
        value = acquisition(X)

        assert value.shape == torch.Size([1])
        assert torch.isfinite(value).all()


def test_reduced_multifidelity_accepts_tensor_like_data_fidelities() -> None:
    train_x, train_y = _data()
    fidelity_dims = torch.tensor([-1], dtype=torch.long)
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        model = model_class(
            train_x,
            train_y,
            n_components=2,
            data_fidelities=fidelity_dims,
            **kwargs,
        )
        assert model.fidelity_dims == (5,)
        assert model.encoded_fidelity_dims == (2,)
        assert torch.isfinite(model.posterior(train_x[:2]).mean).all()


def test_all_reduced_multifidelity_models_preserve_iteration_and_data_roles() -> None:
    torch.manual_seed(0)
    design = torch.rand(10, 4, dtype=torch.double)
    iteration = torch.linspace(0.1, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    data = torch.linspace(0.2, 1.0, 10, dtype=torch.double).unsqueeze(-1)
    train_x = torch.cat((design, iteration, data), dim=-1)
    train_y = design[:, :2].sum(dim=-1, keepdim=True) + iteration + data
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        model = model_class(
            train_x,
            train_y,
            n_components=2,
            iteration_fidelity=-2,
            data_fidelities=[-1],
            **kwargs,
        )
        encoded = model._encode_inputs(train_x)
        assert model.iteration_fidelity == 4
        assert model.data_fidelities == (5,)
        assert model.fidelity_dims == (4, 5)
        assert model.encoded_fidelity_dims == (2, 3)
        torch.testing.assert_close(encoded[..., 2], train_x[..., 4])
        torch.testing.assert_close(encoded[..., 3], train_x[..., 5])


def test_all_reduced_multifidelity_models_reject_overlapping_roles() -> None:
    train_x, train_y = _data()
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        try:
            model_class(
                train_x,
                train_y,
                n_components=2,
                iteration_fidelity=-1,
                data_fidelities=[-1],
                **kwargs,
            )
        except ValueError as error:
            assert "duplicates" in str(error)
        else:
            raise AssertionError(f"Expected {model_class.__name__} to reject overlapping roles.")


def test_all_reduced_multifidelity_models_reject_linear_truncated() -> None:
    train_x, train_y = _data()
    model_classes = (PCAMultiFidelityGP, PLSMultiFidelityGP, RandomProjectionMultiFidelityGP)

    for model_class in model_classes:
        kwargs = {"random_state": 7} if model_class is RandomProjectionMultiFidelityGP else {}
        try:
            model_class(
                train_x,
                train_y,
                n_components=2,
                data_fidelities=[-1],
                linear_truncated=True,
                **kwargs,
            )
        except ValueError as error:
            assert "linear_truncated=False" in str(error)
        else:
            raise AssertionError(f"Expected {model_class.__name__} to reject linear truncation.")

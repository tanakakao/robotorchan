import inspect

import torch
from botorch.models.additive_gp import OrthogonalAdditiveGP as BoTorchOrthogonalAdditiveGP
from botorch.models.contextual import LCEAGP as BoTorchLCEAGP
from botorch.models.contextual import SACGP as BoTorchSACGP
from botorch.models.contextual_multioutput import LCEMGP as BoTorchLCEMGP
from botorch.models.heterogeneous_mtgp import HeterogeneousMTGP as BoTorchHeterogeneousMTGP
from botorch.models.hierarchical.conditional_kernel_gp import (
    HierarchicalConditionalKernelGP as BoTorchHierarchicalConditionalKernelGP,
)
from botorch.models.hierarchical.conditional_kernel_gp import (
    HierarchicalConditionalKernelMultiTaskGP as BoTorchHierarchicalConditionalKernelMultiTaskGP,
)
from botorch.models.map_saas import (
    AdditiveMapSaasSingleTaskGP as BoTorchAdditiveMapSaasSingleTaskGP,
)
from botorch.models.map_saas import (
    EnsembleMapSaasSingleTaskGP as BoTorchEnsembleMapSaasSingleTaskGP,
)
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP as BoTorchRobustRelevancePursuitSingleTaskGP,
)
from gpytorch.mlls import ExactMarginalLogLikelihood

from robotorchan.models import (
    LCEAGP,
    LCEMGP,
    SACGP,
    AdditiveMapSaasSingleTaskGP,
    EnsembleMapSaasSingleTaskGP,
    HeterogeneousMTGP,
    HierarchicalConditionalKernelGP,
    HierarchicalConditionalKernelMultiTaskGP,
    OrthogonalAdditiveGP,
    RobustRelevancePursuitSingleTaskGP,
)


def _assert_constructor_surface(wrapper: type, upstream: type) -> None:
    wrapper_signature = inspect.signature(wrapper.__init__)
    upstream_signature = inspect.signature(upstream.__init__)

    assert tuple(wrapper_signature.parameters) == tuple(upstream_signature.parameters)
    for name in wrapper_signature.parameters:
        assert wrapper_signature.parameters[name].kind == upstream_signature.parameters[name].kind
        assert (
            wrapper_signature.parameters[name].default
            == upstream_signature.parameters[name].default
        )


def _single_task_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_X = torch.tensor(
        [
            [0.05, 0.10],
            [0.15, 0.25],
            [0.30, 0.40],
            [0.45, 0.55],
            [0.60, 0.65],
            [0.75, 0.80],
            [0.85, 0.90],
            [0.95, 0.98],
        ],
        dtype=torch.double,
    )
    train_Y = (torch.sin(train_X[:, :1] * 3.0) + train_X[:, 1:2]).to(torch.double)
    train_Yvar = torch.full_like(train_Y, 1e-4)
    return train_X, train_Y, train_Yvar


def _hierarchical_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[int, dict[int | float, list[int]]],
]:
    train_X = torch.tensor(
        [
            [0.10, 0.20, 0.0],
            [0.20, 0.40, 0.0],
            [0.30, 0.60, 0.0],
            [0.40, 0.80, 0.0],
            [0.60, 0.15, 1.0],
            [0.70, 0.35, 1.0],
            [0.80, 0.55, 1.0],
            [0.90, 0.75, 1.0],
        ],
        dtype=torch.double,
    )
    train_Y = (train_X[:, :1] + 0.25 * train_X[:, 1:2]).to(torch.double)
    dependencies = {2: {0: [1]}}
    return train_X, train_Y, dependencies


def _multitask_hierarchical_data() -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[int, dict[int | float, list[int]]],
]:
    base_X, _, dependencies = _hierarchical_data()
    task_zero = torch.zeros(base_X.shape[0] // 2, 1, dtype=torch.double)
    task_one = torch.ones(base_X.shape[0] // 2, 1, dtype=torch.double)
    tasks = torch.cat([task_zero, task_one], dim=0)
    train_X = torch.cat([base_X, tasks], dim=-1)
    train_Y = train_X[:, :1] + 0.2 * train_X[:, 1:2] + 0.1 * tasks
    return train_X, train_Y, dependencies


def _contextual_data() -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    train_X, train_Y, _ = _single_task_data()
    decomposition = {"context_0": [0], "context_1": [1]}
    return train_X, train_Y, decomposition


def _lcem_data() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0.05, 0.95, 8, dtype=torch.double).unsqueeze(-1)
    tasks = torch.tensor([[0.0], [1.0]] * 4, dtype=torch.double)
    train_X = torch.cat([x, tasks], dim=-1)
    train_Y = torch.sin(x * 2.0) + 0.15 * tasks
    return train_X, train_Y


def _heterogeneous_data() -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
]:
    X0 = torch.tensor(
        [[0.10, 0.20], [0.30, 0.40], [0.55, 0.60], [0.80, 0.90]],
        dtype=torch.double,
    )
    X1 = torch.tensor([[0.15], [0.35], [0.65], [0.85]], dtype=torch.double)
    Y0 = (X0[:, :1] + X0[:, 1:2]).to(torch.double)
    Y1 = (0.5 + X1).to(torch.double)
    return [X0, X1], [Y0, Y1], [[0, 1], [1]]


def test_phase9_wrappers_match_upstream_constructor_surfaces() -> None:
    pairs = [
        (OrthogonalAdditiveGP, BoTorchOrthogonalAdditiveGP),
        (AdditiveMapSaasSingleTaskGP, BoTorchAdditiveMapSaasSingleTaskGP),
        (EnsembleMapSaasSingleTaskGP, BoTorchEnsembleMapSaasSingleTaskGP),
        (
            RobustRelevancePursuitSingleTaskGP,
            BoTorchRobustRelevancePursuitSingleTaskGP,
        ),
        (HierarchicalConditionalKernelGP, BoTorchHierarchicalConditionalKernelGP),
        (
            HierarchicalConditionalKernelMultiTaskGP,
            BoTorchHierarchicalConditionalKernelMultiTaskGP,
        ),
        (HeterogeneousMTGP, BoTorchHeterogeneousMTGP),
        (SACGP, BoTorchSACGP),
        (LCEAGP, BoTorchLCEAGP),
        (LCEMGP, BoTorchLCEMGP),
    ]
    for wrapper, upstream in pairs:
        _assert_constructor_surface(wrapper, upstream)


def test_additive_and_map_saas_wrappers_use_exact_gp_contract() -> None:
    train_X, train_Y, train_Yvar = _single_task_data()
    models = [
        OrthogonalAdditiveGP(train_X=train_X, train_Y=train_Y),
        AdditiveMapSaasSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            num_taus=2,
        ),
        EnsembleMapSaasSingleTaskGP(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            num_taus=2,
            taus=torch.tensor([0.05, 0.10], dtype=torch.double),
        ),
    ]

    for model in models:
        assert model.supports_mll is True
        assert torch.equal(model.raw_train_X, train_X)
        assert torch.equal(model.raw_train_Y, train_Y)
        assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)

    assert models[0].raw_train_Yvar is None
    assert torch.equal(models[1].raw_train_Yvar, train_Yvar)
    assert torch.equal(models[2].raw_train_Yvar, train_Yvar)


def test_robust_wrapper_preserves_fit_dispatch_model_and_raw_data() -> None:
    train_X, train_Y, train_Yvar = _single_task_data()
    model = RobustRelevancePursuitSingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=train_Yvar,
    )

    assert isinstance(model, BoTorchRobustRelevancePursuitSingleTaskGP)
    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert torch.equal(model.raw_train_Yvar, train_Yvar)
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_hierarchical_wrappers_retain_untransformed_training_data() -> None:
    train_X, train_Y, dependencies = _hierarchical_data()
    single = HierarchicalConditionalKernelGP(
        train_X=train_X,
        train_Y=train_Y,
        hierarchical_dependencies=dependencies,
        use_saas_prior=False,
    )

    mt_X, mt_Y, mt_dependencies = _multitask_hierarchical_data()
    multi = HierarchicalConditionalKernelMultiTaskGP(
        train_X=mt_X,
        train_Y=mt_Y,
        task_feature=-1,
        hierarchical_dependencies=mt_dependencies,
        use_saas_prior=False,
    )

    for model, X, Y in [(single, train_X, train_Y), (multi, mt_X, mt_Y)]:
        assert model.supports_mll is True
        assert torch.equal(model.raw_train_X, X)
        assert torch.equal(model.raw_train_Y, Y)
        assert model.raw_train_Yvar is None
        assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_contextual_wrappers_use_common_exact_gp_contract() -> None:
    train_X, train_Y, decomposition = _contextual_data()
    sac = SACGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=None,
        decomposition=decomposition,
    )
    lcea = LCEAGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=None,
        decomposition=decomposition,
    )

    mt_X, mt_Y = _lcem_data()
    lcem = LCEMGP(train_X=mt_X, train_Y=mt_Y, task_feature=-1)

    models_and_data = [
        (sac, train_X, train_Y),
        (lcea, train_X, train_Y),
        (lcem, mt_X, mt_Y),
    ]
    for model, X, Y in models_and_data:
        assert model.supports_mll is True
        assert torch.equal(model.raw_train_X, X)
        assert torch.equal(model.raw_train_Y, Y)
        assert model.raw_train_Yvar is None
        assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)


def test_heterogeneous_wrapper_retains_task_grouped_raw_data() -> None:
    train_Xs, train_Ys, feature_indices = _heterogeneous_data()
    model = HeterogeneousMTGP(
        train_Xs=train_Xs,
        train_Ys=train_Ys,
        train_Yvars=None,
        feature_indices=feature_indices,
        full_feature_dim=2,
        use_saas_prior=False,
    )

    assert isinstance(model, BoTorchHeterogeneousMTGP)
    assert model.supports_mll is True
    assert model.raw_train_Yvars is None
    assert set(model.raw_data) == {"train_Xs", "train_Ys", "train_Yvars"}
    assert isinstance(model.make_mll(), ExactMarginalLogLikelihood)
    for raw, original in zip(model.raw_train_Xs, train_Xs, strict=True):
        assert torch.equal(raw, original)
        assert raw.data_ptr() != original.data_ptr()
    for raw, original in zip(model.raw_train_Ys, train_Ys, strict=True):
        assert torch.equal(raw, original)
        assert raw.data_ptr() != original.data_ptr()


def test_representative_specialized_posteriors_match_upstream() -> None:
    train_X, train_Y, _ = _single_task_data()
    test_X = torch.tensor([[0.20, 0.30], [0.70, 0.80]], dtype=torch.double)

    torch.manual_seed(1234)
    wrapper = OrthogonalAdditiveGP(train_X=train_X, train_Y=train_Y)
    torch.manual_seed(1234)
    upstream = BoTorchOrthogonalAdditiveGP(train_X=train_X, train_Y=train_Y)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)


def test_heterogeneous_posterior_matches_upstream_with_same_initialization() -> None:
    train_Xs, train_Ys, feature_indices = _heterogeneous_data()
    kwargs = {
        "train_Xs": train_Xs,
        "train_Ys": train_Ys,
        "train_Yvars": None,
        "feature_indices": feature_indices,
        "full_feature_dim": 2,
        "use_saas_prior": False,
    }

    torch.manual_seed(1234)
    wrapper = HeterogeneousMTGP(**kwargs)
    torch.manual_seed(1234)
    upstream = BoTorchHeterogeneousMTGP(**kwargs)

    test_X = torch.tensor([[0.25, 0.30], [0.65, 0.75]], dtype=torch.double)
    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)

import inspect

import numpy as np
import torch
from botorch.models import PairwiseGP as BoTorchPairwiseGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize

from robotorchan.models.pairwise import PairwiseGP


def _preference_data() -> tuple[torch.Tensor, torch.Tensor]:
    datapoints = torch.tensor(
        [
            [0.0, 0.0],
            [0.2, 0.8],
            [0.7, 0.3],
            [1.0, 1.0],
        ],
        dtype=torch.double,
    )
    comparisons = torch.tensor([[1, 0], [2, 0], [3, 1], [3, 2]], dtype=torch.long)
    return datapoints, comparisons


def test_pairwise_gp_matches_upstream_constructor_surface() -> None:
    wrapper = inspect.signature(PairwiseGP.__init__)
    upstream = inspect.signature(BoTorchPairwiseGP.__init__)

    assert tuple(wrapper.parameters) == tuple(upstream.parameters)
    for name in wrapper.parameters:
        assert wrapper.parameters[name].kind == upstream.parameters[name].kind
        assert wrapper.parameters[name].default == upstream.parameters[name].default


def test_pairwise_gp_retains_raw_preference_data() -> None:
    datapoints, comparisons = _preference_data()
    original_datapoints = datapoints.clone()
    original_comparisons = comparisons.clone()

    model = PairwiseGP(
        datapoints=datapoints,
        comparisons=comparisons,
        input_transform=Normalize(d=2),
    )

    assert isinstance(model, BoTorchPairwiseGP)
    assert model.supports_mll is True
    assert torch.equal(model.raw_datapoints, original_datapoints)
    assert torch.equal(model.raw_comparisons, original_comparisons)
    assert model.raw_datapoints.data_ptr() != datapoints.data_ptr()
    assert model.raw_comparisons.data_ptr() != comparisons.data_ptr()
    assert set(model.raw_data) == {"datapoints", "comparisons"}
    assert not hasattr(model, "raw_train_Y")


def test_pairwise_gp_raw_data_precedes_duplicate_consolidation() -> None:
    datapoints = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    comparisons = torch.tensor([[0, 2], [1, 2]], dtype=torch.long)

    model = PairwiseGP(datapoints=datapoints, comparisons=comparisons)

    assert model.raw_datapoints.shape[-2] == 3
    assert model.datapoints.shape[-2] == 2
    assert torch.equal(model.raw_comparisons, comparisons)


def test_pairwise_gp_supports_prior_only_inputs() -> None:
    datapoints, _ = _preference_data()

    model = PairwiseGP(datapoints=datapoints, comparisons=None)

    assert torch.equal(model.raw_datapoints, datapoints)
    assert model.raw_comparisons is None

    empty_model = PairwiseGP(datapoints=None, comparisons=None)
    assert empty_model.raw_datapoints is None
    assert empty_model.raw_comparisons is None
    assert set(empty_model.raw_data) == {"datapoints", "comparisons"}


def test_pairwise_gp_make_mll() -> None:
    datapoints, comparisons = _preference_data()
    model = PairwiseGP(datapoints=datapoints, comparisons=comparisons)

    mll = model.make_mll()

    assert isinstance(mll, PairwiseLaplaceMarginalLogLikelihood)
    assert mll.model is model
    assert mll.likelihood is model.likelihood


def test_pairwise_gp_raw_buffers_follow_dtype_and_serialize() -> None:
    datapoints, comparisons = _preference_data()
    model = PairwiseGP(datapoints=datapoints, comparisons=comparisons)

    state_dict = model.state_dict()
    model = model.to(dtype=torch.float32)

    assert "_raw_datapoints" in state_dict
    assert "_raw_comparisons" in state_dict
    assert model.raw_datapoints.dtype == torch.float32
    assert model.raw_comparisons.dtype == torch.long


def test_pairwise_gp_matches_upstream_posterior() -> None:
    datapoints, comparisons = _preference_data()
    test_X = torch.tensor([[0.1, 0.4], [0.8, 0.6]], dtype=torch.double)

    torch.manual_seed(1234)
    np.random.seed(1234)
    wrapper = PairwiseGP(datapoints=datapoints, comparisons=comparisons)

    torch.manual_seed(1234)
    np.random.seed(1234)
    upstream = BoTorchPairwiseGP(datapoints=datapoints, comparisons=comparisons)

    wrapper.eval()
    upstream.eval()
    wrapper_posterior = wrapper.posterior(test_X)
    upstream_posterior = upstream.posterior(test_X)

    torch.testing.assert_close(wrapper_posterior.mean, upstream_posterior.mean)
    torch.testing.assert_close(wrapper_posterior.variance, upstream_posterior.variance)

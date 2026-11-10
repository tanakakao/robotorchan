"""Joint mixed EUBO proposes valid pairs with independently chosen categories."""

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption

from robotorchan.models import MixedPairwiseGP
from robotorchan.optim.preference import optimize_mixed_preference_pair


def test_mixed_eubo_joint_pair() -> None:
    torch.manual_seed(0)
    datapoints = torch.tensor(
        [[0.0, 0.0], [0.3, 1.0], [0.7, 0.0], [1.0, 1.0]],
        dtype=torch.double,
    )
    comparisons = torch.tensor([[1, 0], [2, 3], [1, 2]], dtype=torch.long)
    model = MixedPairwiseGP(datapoints, comparisons, cat_dims=[1])
    model.fit(optimizer_kwargs={"options": {"maxiter": 10}})
    model.eval()
    acquisition = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    pair, value = optimize_mixed_preference_pair(
        acquisition,
        bounds,
        fixed_features_list=[{1: 0.0}, {1: 1.0}],
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 10},
    )
    assert pair.shape == (2, 2)
    assert torch.isfinite(pair).all()
    assert torch.isfinite(value)
    assert ((pair >= bounds[0]) & (pair <= bounds[1])).all()
    assert set(pair[:, 1].tolist()) <= {0.0, 1.0}

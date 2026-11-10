"""BoTorch-native EUBO integration with the robotorchan preference model."""

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

from robotorchan.models import PairwiseGP


def test_pairwise_gp_analytic_eubo_proposes_a_comparison_pair() -> None:
    torch.manual_seed(0)
    datapoints = torch.linspace(0.0, 1.0, 6, dtype=torch.double).unsqueeze(-1)
    comparisons = torch.tensor([[1, 0], [2, 1], [2, 3], [3, 4], [4, 5]], dtype=torch.long)
    model = PairwiseGP(datapoints=datapoints, comparisons=comparisons)
    model.fit(optimizer_kwargs={"options": {"maxiter": 10}})
    model.eval()

    acquisition = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=datapoints.dtype)
    candidates, value = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=2,
        num_restarts=2,
        raw_samples=16,
        options={"maxiter": 20},
    )

    assert candidates.shape == (2, 1)
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((bounds[0] <= candidates) & (candidates <= bounds[1])).all()

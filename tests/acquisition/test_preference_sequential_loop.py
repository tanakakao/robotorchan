"""End-to-end sequential preference BO with BoTorch-native EUBO."""

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

from robotorchan.models import PairwiseGP


def test_sequential_preference_eubo_loop() -> None:
    torch.manual_seed(0)
    datapoints = torch.tensor([[0.0], [0.4], [0.8], [1.0]], dtype=torch.double)
    comparisons = torch.tensor([[1, 0], [2, 3], [1, 2]], dtype=torch.long)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    for _ in range(2):
        model = PairwiseGP(datapoints, comparisons)
        model.fit(optimizer_kwargs={"options": {"maxiter": 10}})
        model.eval()

        acquisition = AnalyticExpectedUtilityOfBestOption(pref_model=model)
        pair, value = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=2,
            num_restarts=2,
            raw_samples=16,
            options={"maxiter": 20},
        )
        assert pair.shape == (2, 1)
        assert torch.isfinite(pair).all()
        assert torch.isfinite(value).all()

        # Simulate the user's pairwise feedback, using a hidden utility.
        utility = torch.sin(2 * torch.pi * pair.squeeze(-1))
        winner, loser = (0, 1) if utility[0] >= utility[1] else (1, 0)
        first_new_index = datapoints.shape[0]
        datapoints = torch.cat((datapoints, pair.detach()), dim=0)
        comparison = torch.tensor(
            [[first_new_index + winner, first_new_index + loser]], dtype=torch.long
        )
        comparisons = torch.cat((comparisons, comparison), dim=0)

    # Refitting on the complete accumulated history closes the loop.
    updated_model = PairwiseGP(datapoints, comparisons)
    updated_model.fit(optimizer_kwargs={"options": {"maxiter": 10}})
    updated_model.eval()

    assert datapoints.shape == (8, 1)
    assert comparisons.shape == (5, 2)
    assert comparisons.max().item() < datapoints.shape[0]
    assert torch.isfinite(updated_model.posterior(datapoints).mean).all()

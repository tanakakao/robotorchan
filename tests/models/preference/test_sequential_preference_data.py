"""Focused sequential preference-data workflow regression test."""

import torch

from robotorchan.models import PairwiseGP


def test_sequential_preference_data_rebuild_and_refit() -> None:
    datapoints = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
    comparisons = torch.tensor([[1, 0], [2, 1]], dtype=torch.long)

    model = PairwiseGP(datapoints, comparisons)
    model.fit(optimizer_kwargs={"options": {"maxiter": 10}})

    # One additional comparison of existing points and one new candidate.
    new_datapoints = torch.cat((datapoints, torch.tensor([[0.75]], dtype=datapoints.dtype)), dim=0)
    new_comparisons = torch.cat(
        (
            comparisons,
            torch.tensor([[2, 0], [3, 1]], dtype=torch.long),
        ),
        dim=0,
    )

    assert new_comparisons.dtype == torch.long
    assert new_comparisons.min().item() >= 0
    assert new_comparisons.max().item() < len(new_datapoints)
    assert torch.all(new_comparisons[:, 0] != new_comparisons[:, 1])

    updated = PairwiseGP(new_datapoints, new_comparisons)
    updated.fit(optimizer_kwargs={"options": {"maxiter": 10}})

    assert torch.equal(updated.raw_datapoints, new_datapoints)
    assert torch.equal(updated.raw_comparisons, new_comparisons)
    updated.eval()
    posterior = updated.posterior(new_datapoints)
    assert posterior.mean.shape[-2] == len(new_datapoints)
    assert torch.isfinite(posterior.mean).all()

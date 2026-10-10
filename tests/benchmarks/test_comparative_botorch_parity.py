"""Tests for deterministic BoTorch benchmark adapter parity."""

from unittest.mock import patch

import pytest
import torch

from robotorchan.benchmarks.comparative_botorch_parity import compare_botorch_candidates
from robotorchan.benchmarks.standard_problems import branin


@pytest.mark.parametrize("acquisition", ["qEI", "qNEI"])
def test_adapter_matches_native_candidate_generator(acquisition: str) -> None:
    problem = branin()
    X = torch.tensor([[-3.0, 4.0], [0.0, 7.0], [3.0, 11.0]], dtype=torch.double)
    Y = problem.evaluate_observation(X)
    calls = []

    def deterministic_native(problem, X, Y, q, generator, **kwargs):
        del problem, Y
        calls.append(kwargs["acquisition"])
        return torch.rand((q, X.shape[-1]), generator=generator, dtype=X.dtype)

    with (
        patch(
            "robotorchan.benchmarks.comparative_botorch_parity.botorch_gp_candidates",
            side_effect=deterministic_native,
        ),
        patch(
            "robotorchan.benchmarks.comparative_qei.botorch_gp_candidates",
            side_effect=deterministic_native,
        ),
        patch(
            "robotorchan.benchmarks.comparative_qnei.botorch_gp_candidates",
            side_effect=deterministic_native,
        ),
    ):
        result = compare_botorch_candidates(problem, X, Y, acquisition=acquisition)
    assert calls == [acquisition, acquisition]
    assert result.within_tolerance
    assert result.max_absolute_difference == 0.0


def test_unknown_acquisition_rejected() -> None:
    problem = branin()
    X = torch.zeros((3, 2), dtype=torch.double)
    with pytest.raises(ValueError, match="qEI and qNEI"):
        compare_botorch_candidates(problem, X, torch.zeros((3, 1)), acquisition="qUCB")

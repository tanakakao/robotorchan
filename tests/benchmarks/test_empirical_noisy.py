"""Validate noisy empirical cohort input and provenance contracts."""

import pytest
import torch

from robotorchan.benchmarks.empirical_noisy import run_noisy_singleobjective
from robotorchan.benchmarks.observation_noise import SeededObservationNoise


@pytest.mark.parametrize(
    ("problem", "q", "std", "policy"),
    [
        ("unknown", 1, 0.1, "paired"),
        ("branin", 2, 0.1, "paired"),
        ("branin", 1, 0.0, "paired"),
        ("branin", 1, float("nan"), "paired"),
        ("branin", 1, 0.1, "unsupported"),
    ],
)
def test_noisy_empirical_rejects_invalid_cohorts(tmp_path, problem, q, std, policy):
    with pytest.raises(ValueError):
        run_noisy_singleobjective(
            tmp_path / "invalid",
            problem=problem,
            q=q,
            git_sha="abc123",
            dirty=False,
            noise_std=std,
            noise_policy=policy,
        )


def test_noisy_empirical_refuses_existing_artifacts(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_noisy_singleobjective(
            destination,
            problem="branin",
            q=1,
            git_sha="abc123",
            dirty=False,
        )


def test_seeded_gaussian_noise_pairing_is_repeatable():
    def gaussian(truth, generator):
        return torch.randn(truth.shape, dtype=truth.dtype, generator=generator) * 0.1

    policy = SeededObservationNoise(gaussian, policy="paired")
    truth = torch.zeros(20, 1, dtype=torch.double)
    a = policy.observe(truth, policy.generator(2, "random", truth.device))
    b = policy.observe(truth, policy.generator(2, "qNEI", truth.device))
    assert torch.equal(a, b)
    assert not torch.equal(a, truth)

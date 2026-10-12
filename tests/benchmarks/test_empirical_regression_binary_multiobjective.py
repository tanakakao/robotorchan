"""Verify feasible hypervolume and canonical binary labels."""

import torch

from robotorchan.benchmarks.constrained_evidence import observed_pass_labels
from robotorchan.benchmarks.heterogeneous_problems import strength_conductivity_pass
from robotorchan.benchmarks.metrics import hypervolume_2d


def test_multiobjective_pass_labels_filter_hypervolume():
    problem = strength_conductivity_pass()
    X = torch.tensor([[0.8, 0.0], [0.2, 0.8]], dtype=torch.float64)
    labels = observed_pass_labels(problem.name, X).squeeze(-1).bool()
    values = problem.evaluate_truth(X)
    assert labels.tolist() == [False, True]
    assert torch.equal(labels.unsqueeze(-1), problem.evaluate_constraints(X) >= 0)
    reference = problem.reference_point
    assert hypervolume_2d(values[~labels], reference) > 0
    assert torch.equal(
        hypervolume_2d(values[labels], reference),
        hypervolume_2d(values[labels], reference),
    )
    assert hypervolume_2d(values[:0], reference).item() == 0.0

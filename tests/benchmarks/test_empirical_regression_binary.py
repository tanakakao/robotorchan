"""Validate canonical observed labels and feasible regression evidence."""

import torch

from robotorchan.benchmarks.constrained_evidence import observed_pass_labels
from robotorchan.benchmarks.regression_binary_problems import strength_pass


def test_regression_binary_labels_are_distinct_from_constraint_margins():
    problem = strength_pass()
    X = torch.tensor([[0.0, 0.0], [0.0, 0.25], [0.8, 0.9]], dtype=torch.float64)
    labels = observed_pass_labels("strength_pass", X)
    margins = problem.evaluate_constraints(X)
    assert labels.shape == margins.shape == (3, 1)
    assert labels.squeeze(-1).tolist() == [0.0, 1.0, 1.0]
    assert torch.equal(labels.bool(), margins >= 0)
    assert not torch.equal(labels, margins)

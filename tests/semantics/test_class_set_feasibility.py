"""Class-set feasibility regression tests."""

import pytest
import torch
from torch import nn

from robotorchan.semantics.probability import ClassificationProbabilityOfFeasibility


class _Classifier(nn.Module):
    num_classes = 4

    def predict_proba(self, X):
        return X.new_tensor([0.1, 0.2, 0.3, 0.4]).expand(*X.shape[:-1], 4)


def test_single_class_backward_compatibility():
    X = torch.zeros(3, 1, 2)
    pof = ClassificationProbabilityOfFeasibility(_Classifier(), feasible_class=2)
    torch.testing.assert_close(pof(X), torch.full((3, 1), 0.3))


def test_class_set_probability_and_grad_shape():
    X = torch.zeros(2, 3, 2)
    pof = ClassificationProbabilityOfFeasibility(_Classifier(), feasible_classes=(0, 2, 3))
    torch.testing.assert_close(pof(X), torch.full((2, 3), 0.8))


@pytest.mark.parametrize("classes", [(), (1, 1), (-1,), (4,), (True,)])
def test_invalid_class_sets(classes):
    with pytest.raises(ValueError):
        ClassificationProbabilityOfFeasibility(_Classifier(), feasible_classes=classes)

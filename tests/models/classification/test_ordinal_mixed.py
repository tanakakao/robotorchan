"""Mixed-input and q-batch contracts for ordinal Gaussian processes."""

import torch

from robotorchan.models.capabilities import InputType
from robotorchan.models.classification import (
    MixedOrdinalSingleTaskGPClassifier,
    OrdinalSingleTaskGPClassifier,
)
from robotorchan.models.classification.registry import get_classification_model_entry


def _training_data() -> tuple[torch.Tensor, torch.Tensor]:
    continuous = torch.linspace(0, 1, 12, dtype=torch.double)
    categorical = (torch.arange(12) % 2).to(dtype=torch.double)
    train_X = torch.stack((continuous, categorical), dim=-1)
    train_Y = torch.bucketize(continuous + 0.2 * categorical, torch.tensor([0.4, 0.8], dtype=torch.double))
    return train_X, train_Y


def test_mixed_ordinal_fit_and_probabilities() -> None:
    train_X, train_Y = _training_data()
    model = MixedOrdinalSingleTaskGPClassifier(
        train_X, train_Y, num_classes=3, cat_dims=[1]
    )
    assert model.cat_dims == (1,)
    assert get_classification_model_entry("ordinal.mixed").model_class is type(model)
    assert (
        get_classification_model_entry("ordinal.mixed").capabilities.input_type
        is InputType.MIXED
    )
    model.fit(epochs=2, lr=0.01)
    test_X = train_X[:3]
    probabilities = model.predict_proba(test_X)
    assert probabilities.shape == (3, 3)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(3, dtype=torch.double))
    assert model.sample_class_probabilities(test_X, sample_shape=torch.Size([4])).shape == (4, 3, 3)


def test_ordinal_q_batch_probabilities_preserve_shape() -> None:
    train_X, train_Y = _training_data()
    for model in (
        OrdinalSingleTaskGPClassifier(train_X[:, :1], train_Y, num_classes=3),
        MixedOrdinalSingleTaskGPClassifier(
            train_X, train_Y, num_classes=3, cat_dims=[1]
        ),
    ):
        test_X = train_X[:6, : model.raw_train_X.shape[-1]].reshape(
            2, 3, model.raw_train_X.shape[-1]
        )
        probabilities = model.predict_proba(test_X)
        assert probabilities.shape == (2, 3, 3)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3, dtype=torch.double))
        samples = model.sample_class_probabilities(test_X, sample_shape=torch.Size([4]))
        assert samples.shape == (4, 2, 3, 3)

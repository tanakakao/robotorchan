"""Unified factory construction and public registry coverage."""

import inspect

import pytest
import torch

from robotorchan import models
from robotorchan.models import classification, create_model, get_model_class, list_models
from robotorchan.models.registry import MODEL_REGISTRY


def test_registry_resolution() -> None:
    names = list_models()
    assert names == tuple(sorted(set(names)))
    for name in MODEL_REGISTRY:
        assert get_model_class(name) is getattr(models, name)
    for name, entry in classification.CLASSIFICATION_MODEL_REGISTRY.items():
        assert get_model_class(name) is entry.model_class
        assert get_model_class(entry.model_class.__name__) is entry.model_class
    for name in classification.__all__:
        if name.endswith(("GPClassifier", "ClassificationEnsemble")):
            cls = getattr(classification, name)
            if inspect.isabstract(cls):
                assert name not in names
            else:
                assert get_model_class(name) is cls


def test_special_surrogates() -> None:
    from robotorchan.models.standard import BOCSSurrogate, COMBOSurrogate
    from robotorchan.optim.transfer.rgpe_model import RGPEModel

    for cls in (BOCSSurrogate, COMBOSurrogate, RGPEModel):
        assert cls.__name__ in list_models()
        assert get_model_class(cls.__name__) is cls


def test_non_models_and_unknown_names() -> None:
    for name in (
        "ModelCapabilities",
        "OrderedProbitLikelihood",
        "BootstrapBinaryClassificationEnsemble",
        "not_a_model",
    ):
        assert name not in list_models()
        with pytest.raises(KeyError, match="Unknown model"):
            get_model_class(name)
    with pytest.raises(TypeError, match="registered name"):
        create_model(None)


def test_callable_arguments_and_errors() -> None:
    sentinel = object()

    def constructor(value, *, model):
        assert value is sentinel
        assert model is sentinel
        return sentinel

    assert create_model(constructor, sentinel, model=sentinel) is sentinel
    with pytest.raises(TypeError):
        create_model(constructor, sentinel)


def test_regression_and_composition() -> None:
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    Y = X.sin()
    model = create_model("SingleTaskGP", train_X=X, train_Y=Y)
    assert type(model) is models.SingleTaskGP
    assert model.posterior(X).mean.shape == Y.shape
    direct = create_model(models.SingleTaskGP, X, Y)
    assert type(direct) is type(model)
    combined = create_model("ModelListGP", model, direct)
    assert type(combined) is models.ModelListGP


@pytest.mark.parametrize(
    ("name", "classes"),
    [
        ("binary.standard", 2),
        ("MulticlassSingleTaskGPClassifier", 3),
        ("OrdinalSingleTaskGPClassifier", 3),
    ],
)
def test_classification_construction(name: str, classes: int) -> None:
    X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
    Y = torch.arange(6) % classes
    kwargs = {"num_classes": classes} if classes > 2 else {}
    model = create_model(name, train_X=X, train_Y=Y, **kwargs)
    assert type(model) is get_model_class(name)
    assert model.num_classes == classes
    assert model.predict_proba(X).shape == (6, classes)

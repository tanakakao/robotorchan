import torch
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)

from robotorchan.models import SingleTaskGP


def _model() -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y)


def test_native_mes_acquisitions_are_compatible() -> None:
    model = _model()
    candidate_set = torch.linspace(0.0, 1.0, 32, dtype=torch.double).unsqueeze(-1)
    X = torch.tensor([[[0.4]]], dtype=torch.double)

    mes = qMaxValueEntropy(model=model, candidate_set=candidate_set, num_mv_samples=4)
    gibbon = qLowerBoundMaxValueEntropy(
        model=model,
        candidate_set=candidate_set,
        num_mv_samples=4,
    )

    assert torch.isfinite(mes(X)).all()
    assert torch.isfinite(gibbon(X)).all()

"""Classification acquisition contract for hybrid DE and BoTorch search."""

import torch

from robotorchan.acquisition.active_learning.classification import PredictiveEntropy
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
)
from robotorchan.optim.backends import optimize_acqf_hybrid


def test_binary_classification_hybrid_de_q_one() -> None:
    torch.manual_seed(0)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x.squeeze(-1) > 0.5).to(dtype=torch.double)
    model = BinarySingleTaskGPClassifier(train_x, train_y, inducing_points=4)
    model.eval()
    acquisition = PredictiveEntropy(model)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    candidates, value = optimize_acqf_hybrid(
        acquisition,
        bounds,
        q=1,
        global_optimizer="de",
        global_options={"options": {"maxiter": 2, "popsize": 5, "polish": False}},
        local_options={"maxiter": 10},
        seed=7,
    )

    assert candidates.shape == torch.Size([1, 1])
    assert torch.isfinite(candidates).all()
    assert torch.isfinite(value).all()
    assert ((candidates >= bounds[0]) & (candidates <= bounds[1])).all()

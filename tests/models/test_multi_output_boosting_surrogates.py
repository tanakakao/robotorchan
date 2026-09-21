import pytest
import torch
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning

pytest.importorskip("sklearn")

from robotorchan.models import GradientBoostingSurrogate, HistGradientBoostingSurrogate


@pytest.mark.parametrize(
    "model_class",
    [GradientBoostingSurrogate, HistGradientBoostingSurrogate],
)
def test_boosting_surrogates_support_multi_output(model_class) -> None:
    train_X = torch.linspace(0.0, 1.0, 28, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        [torch.sin(train_X * 5.0), torch.cos(train_X * 4.0)],
        dim=-1,
    )
    model = model_class(train_X, train_Y, n_members=5, random_state=4)
    model.fit()

    X = torch.tensor([[0.2], [0.7]], dtype=torch.double)
    posterior = model.posterior(X)

    assert model.num_outputs == 2
    assert posterior.values.shape == torch.Size([5, 2, 2])
    assert torch.isfinite(posterior.values).all()
    selected = model.posterior(X, output_indices=[1])
    assert selected.values.shape == torch.Size([5, 2, 1])


@pytest.mark.parametrize(
    "model_class",
    [GradientBoostingSurrogate, HistGradientBoostingSurrogate],
)
def test_multi_output_boosting_integrates_with_qehvi(model_class) -> None:
    train_X = torch.linspace(0.0, 1.0, 28, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat(
        [torch.sin(train_X * 5.0), torch.cos(train_X * 4.0)],
        dim=-1,
    )
    model = model_class(train_X, train_Y, n_members=6, random_state=5)
    model.fit()
    ref_point = train_Y.min(dim=0).values - 0.1
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
    acqf = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning,
    )

    value = acqf(torch.tensor([[[0.25], [0.75]]], dtype=torch.double))

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_multi_output_boosting_rejects_invalid_output_index() -> None:
    train_X = torch.linspace(0.0, 1.0, 20, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.cat([train_X, train_X.square()], dim=-1)
    model = GradientBoostingSurrogate(train_X, train_Y, n_members=3, random_state=1)
    model.fit()

    with pytest.raises(ValueError, match="output_indices"):
        model.posterior(train_X[:2], output_indices=[2])

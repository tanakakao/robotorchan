import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.sampling.normal import SobolQMCNormalSampler

from robotorchan.models import SingleTaskGP


def _model() -> SingleTaskGP:
    train_X = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    train_Y = torch.sin(train_X * 5.0)
    return SingleTaskGP(train_X, train_Y)


def test_native_qknowledge_gradient_is_compatible() -> None:
    model = _model()
    acquisition = qKnowledgeGradient(model=model, num_fantasies=4)
    X = torch.tensor([[[0.4], [0.2], [0.5], [0.8], [0.9]]], dtype=torch.double)

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()


def test_native_qmulti_step_lookahead_is_compatible() -> None:
    model = _model()
    samplers = [
        SobolQMCNormalSampler(sample_shape=torch.Size([2])),
        SobolQMCNormalSampler(sample_shape=torch.Size([2])),
    ]
    acquisition = qMultiStepLookahead(
        model=model,
        batch_sizes=[1, 1],
        samplers=samplers,
    )
    X = torch.rand(
        1,
        acquisition.get_augmented_q_batch_size(q=1),
        1,
        dtype=torch.double,
    )

    value = acquisition(X)

    assert value.shape == torch.Size([1])
    assert torch.isfinite(value).all()

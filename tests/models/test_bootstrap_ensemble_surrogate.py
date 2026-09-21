from typing import Any

import torch

from robotorchan.models.non_gp.bootstrap import BootstrapEnsembleSurrogate


class MeanEstimator:
    def fit(self, X: Any, y: Any) -> None:
        del X
        self.mean = float(y.mean())

    def predict(self, X: Any) -> Any:
        import numpy as np

        return np.full(len(X), self.mean)


class DummyBootstrapSurrogate(BootstrapEnsembleSurrogate):
    model_name = "DummyBootstrapSurrogate"

    def _make_estimator(self, member_index: int) -> MeanEstimator:
        del member_index
        return MeanEstimator()


def test_bootstrap_base_preserves_raw_data_and_member_count() -> None:
    train_X = torch.arange(12, dtype=torch.double).reshape(6, 2)
    train_Y = torch.arange(6, dtype=torch.double).unsqueeze(-1)
    model = DummyBootstrapSurrogate(train_X, train_Y, n_members=4, random_state=3)

    model.fit()

    assert torch.equal(model.raw_train_X, train_X)
    assert torch.equal(model.raw_train_Y, train_Y)
    assert len(model._members) == 4
    assert model.is_fitted


def test_bootstrap_base_is_reproducible_for_fixed_seed() -> None:
    train_X = torch.arange(20, dtype=torch.double).reshape(10, 2)
    train_Y = torch.arange(10, dtype=torch.double).unsqueeze(-1)
    first = DummyBootstrapSurrogate(train_X, train_Y, n_members=5, random_state=7)
    second = DummyBootstrapSurrogate(train_X, train_Y, n_members=5, random_state=7)
    first.fit()
    second.fit()

    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)

    assert torch.equal(first.posterior(X).values, second.posterior(X).values)


def test_bootstrap_base_members_can_express_uncertainty() -> None:
    train_X = torch.arange(20, dtype=torch.double).reshape(10, 2)
    train_Y = torch.arange(10, dtype=torch.double).unsqueeze(-1)
    model = DummyBootstrapSurrogate(train_X, train_Y, n_members=8, random_state=9)
    model.fit()

    posterior = model.posterior(torch.tensor([[1.0, 2.0]], dtype=torch.double))

    assert posterior.values.shape == torch.Size([8, 1, 1])
    assert posterior.values.std() > 0

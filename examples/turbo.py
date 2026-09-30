"""Minimal stateful TuRBO loop using the public strategy API."""

import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll

from robotorchan.models import SingleTaskGP
from robotorchan.optim import TuRBOState, TuRBOStrategy


def objective(X: torch.Tensor) -> torch.Tensor:
    """Simple maximization objective on the unit box."""
    return -((X - 0.7) ** 2).sum(dim=-1, keepdim=True)


def main() -> None:
    torch.set_default_dtype(torch.double)
    input_dim = 5
    bounds = torch.stack([torch.zeros(input_dim), torch.ones(input_dim)])
    sobol = torch.quasirandom.SobolEngine(input_dim, scramble=True, seed=7)
    train_X = sobol.draw(10)
    train_Y = objective(train_X)
    best_index = int(train_Y.squeeze(-1).argmax())
    best_value = float(train_Y[best_index].item())

    strategy = TuRBOStrategy(
        bounds,
        center=train_X[best_index],
        state=TuRBOState(
            dim=input_dim,
            best_value=best_value,
            observed_best_value=best_value,
        ),
        num_restarts=4,
        raw_samples=64,
        seed=7,
    )

    for _ in range(5):
        model = SingleTaskGP(train_X, train_Y)
        fit_gpytorch_mll(model.make_mll())
        acquisition = LogExpectedImprovement(model, best_f=float(train_Y.max().item()))
        result = strategy.optimize(acquisition)
        new_X = result.candidates.detach()
        new_Y = objective(new_X)

        train_X = torch.cat([train_X, new_X])
        train_Y = torch.cat([train_Y, new_Y])
        strategy.update_state(new_Y, candidates=new_X)
        if strategy.state.restart_triggered:
            strategy.restart()

    print("best observed:", float(train_Y.max().item()))
    print("trust-region length:", strategy.state.length)


if __name__ == "__main__":
    main()

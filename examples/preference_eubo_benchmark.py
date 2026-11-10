"""Compare sequential EUBO and random pair selection on synthetic preferences.

Run: python examples/preference_eubo_benchmark.py --trials 3 --rounds 4
This is an empirical demonstration, not a proof of EUBO superiority.
"""

import argparse

import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf

from robotorchan.models import PairwiseGP


def utility(x: torch.Tensor) -> torch.Tensor:
    z = x.squeeze(-1)
    return torch.sin(2 * torch.pi * z) + 0.35 * z


def compare_pair(pair: torch.Tensor, offset: int) -> torch.Tensor:
    values = utility(pair)
    winner, loser = (0, 1) if values[0] >= values[1] else (1, 0)
    return torch.tensor([[offset + winner, offset + loser]], dtype=torch.long)


def run_trial(*, method: str, seed: int, rounds: int) -> list[float]:
    torch.manual_seed(seed)
    datapoints = torch.tensor([[0.0], [0.4], [0.8], [1.0]], dtype=torch.double)
    comparisons = torch.tensor([[1, 0], [2, 3], [1, 2]], dtype=torch.long)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    optimum = utility(torch.linspace(0, 1, 2001, dtype=torch.double).unsqueeze(-1)).max()
    regrets = [float(optimum - utility(datapoints).max())]

    for _ in range(rounds):
        if method == "eubo":
            model = PairwiseGP(datapoints, comparisons)
            model.fit(optimizer_kwargs={"options": {"maxiter": 20}})
            model.eval()
            acquisition = AnalyticExpectedUtilityOfBestOption(pref_model=model)
            pair, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=2,
                num_restarts=3,
                raw_samples=32,
                options={"maxiter": 40},
            )
            pair = pair.detach()
        elif method == "random":
            pair = torch.rand(2, 1, dtype=torch.double)
        else:
            raise ValueError(f"Unknown method: {method}")

        new_comparison = compare_pair(pair, len(datapoints))
        datapoints = torch.cat((datapoints, pair), dim=0)
        comparisons = torch.cat((comparisons, new_comparison), dim=0)
        regrets.append(float(optimum - utility(datapoints).max()))

    return regrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.trials < 1 or args.rounds < 1:
        parser.error("--trials and --rounds must be positive")

    for method in ("eubo", "random"):
        curves = torch.tensor(
            [
                run_trial(method=method, seed=args.seed + trial, rounds=args.rounds)
                for trial in range(args.trials)
            ],
            dtype=torch.double,
        )
        mean = curves.mean(dim=0)
        print(f"{method}: mean simple regret by round = {[round(v, 5) for v in mean.tolist()]}")
        print(f"{method}: final regret mean={mean[-1]:.5f}, std={curves[:, -1].std(unbiased=False):.5f}")


if __name__ == "__main__":
    main()

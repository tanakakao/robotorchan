"""Summarize paired benchmark results at completed evaluation budgets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import torch

from robotorchan.benchmarks.multiseed_statistics import compare_paired_bootstrap

# Same problem/evaluation contract, with mode and library added for a comparison cell.
Contract = tuple[str, int, int, int]
Key = tuple[Contract, str, str]


def identify(row: dict) -> Key:
    config = row["config"]
    contract = (
        config["problem"],
        config["q"],
        config["initial_points"],
        row["evaluation_budget_total"],
    )
    return contract, row["mode"], row["library"]


def group_records(rows: list[dict]) -> dict[Key, dict[int, dict]]:
    """Reject duplicate trials, mixed software settings and mismatched initial designs."""
    groups: dict[Key, dict[int, dict]] = defaultdict(dict)
    settings = set()
    library_versions: dict[str, set] = defaultdict(set)
    paired_ids = defaultdict(set)
    for row in rows:
        if row["schema_version"] != 1:
            raise ValueError("Unsupported result schema version.")
        settings.add(
            (row["commit"], row["profile"], json.dumps(row["environment"], sort_keys=True))
        )
        library_versions[row["library"]].add(row["library_version"])
        key, seed = identify(row), row["seed"]
        if seed in groups[key]:
            raise ValueError(f"Duplicate trial for {key}, seed {seed}.")
        groups[key][seed] = row
        if row["status"] == "ok":
            if row["evaluation_count"] != key[0][3]:
                raise ValueError(f"Evaluation budget mismatch for {key}, seed {seed}.")
            if not row["initial_data_id"]:
                raise ValueError(f"Missing initial design ID for {key}, seed {seed}.")
            paired_ids[key[0], seed].add(row["initial_data_id"])
    if len(settings) > 1 or any(len(items) != 1 for items in library_versions.values()):
        raise ValueError("Analyze one software environment and settings profile at a time.")
    if any(len(ids) > 1 for ids in paired_ids.values()):
        raise ValueError("Paired trials disagree on the initial data fingerprint.")
    return dict(groups)


def metadata(key: Key) -> dict:
    (problem, q, initial, budget), mode, library = key
    return {
        "problem": problem,
        "q": q,
        "initial": initial,
        "budget": budget,
        "mode": mode,
        "library": library,
    }


def measures(rows: dict[int, dict]) -> tuple[str, dict[int, list]]:
    successful = {seed: row["metrics"] for seed, row in rows.items() if row["status"] == "ok"}
    for metric in ("simple_regret", "hypervolume"):
        if any(metric in result for result in successful.values()):
            return metric, {
                seed: result[metric] for seed, result in successful.items() if metric in result
            }
    return "", {}


def quantiles(values: list[float]) -> tuple:
    if not values:
        return None, None, None
    low, med, high = np.quantile(values, (0.25, 0.5, 0.75))
    return float(low), float(med), float(high)


def valid(value) -> bool:
    return value is not None and isfinite(value)


def compare(groups: dict[Key, dict[int, dict]], expected: set[int]) -> tuple[list, list, list]:
    summary, paired, curves = [], [], []
    measurements = {key: measures(rows) for key, rows in groups.items()}
    for key, rows in sorted(groups.items()):
        meta = metadata(key)
        successful = [row for row in rows.values() if row["status"] == "ok"]
        metric, per_seed = measurements[key]
        if set(rows) - expected:
            raise ValueError(f"Unexpected seed for {key}.")
        finals = [curve[-1] for curve in per_seed.values() if valid(curve[-1])]
        lower, med, upper = quantiles(finals)
        summary.append(
            meta
            | {
                "metric": metric,
                "expected_runs": len(expected),
                "recorded_runs": len(rows),
                "successes": len(successful),
                "failures": len(rows) - len(successful),
                "missing_runs": len(expected - rows.keys()),
                "finite_final_runs": len(finals),
                "mean_final": mean(finals) if finals else None,
                "sd_final": stdev(finals) if len(finals) > 1 else None,
                "p25_final": lower,
                "median_final": med,
                "p75_final": upper,
                "feasible_run_count": sum(
                    row["metrics"]["feasible_count"] > 0 for row in successful
                ),
                "median_violation_rate": median(
                    row["metrics"]["constraint_violation_rate"] for row in successful
                )
                if successful
                else None,
                "median_candidate_wall_seconds": median(
                    sum(row["trajectory"]["candidate_seconds"]) for row in successful
                )
                if successful
                else None,
            }
        )
        if not per_seed:
            continue
        start, q, total = meta["initial"], meta["q"], meta["budget"]
        # A q>1 batch becomes visible only after all its observations have returned.
        for budget in sorted(set(range(start, total + 1, q)) | {total}):
            values = [curve[budget - 1] for curve in per_seed.values() if valid(curve[budget - 1])]
            p25, p50, p75 = quantiles(values)
            curves.append(
                meta
                | {
                    "metric": metric,
                    "evaluations": budget,
                    "finite_runs": len(values),
                    "p25": p25,
                    "median": p50,
                    "p75": p75,
                }
            )
        contract, mode, _ = key
        reference_mode, reference_library = {
            "A": ("A", "sobol"),
            "B": ("B", "botorch"),
            "C": ("A", "optuna"),
        }[mode]
        ref_key = (contract, reference_mode, reference_library)
        if ref_key == key or ref_key not in measurements:
            continue
        ref_metric, reference = measurements[ref_key]
        if ref_metric != metric:
            continue
        ids = sorted(
            seed
            for seed in per_seed.keys() & reference.keys()
            if valid(per_seed[seed][-1]) and valid(reference[seed][-1])
        )
        if not ids:
            continue
        differences = [per_seed[seed][-1] - reference[seed][-1] for seed in ids]
        ci = {}
        if len(ids) >= 2:
            result = compare_paired_bootstrap(
                {seed: torch.tensor([per_seed[seed][-1]], dtype=torch.float64) for seed in ids},
                {seed: torch.tensor([reference[seed][-1]], dtype=torch.float64) for seed in ids},
                n_resamples=10_000,
            ).difference
            ci = {"ci_lower": result.lower.item(), "ci_upper": result.upper.item()}
        paired.append(
            meta
            | {
                "metric": metric,
                "reference": reference_library,
                "reference_mode": reference_mode,
                "pairs": len(ids),
                "mean_difference": mean(differences),
                "ci_lower": None,
                "ci_upper": None,
                **ci,
            }
        )
    return summary, paired, curves


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_curves(curves: list[dict], destination: Path) -> None:
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for row in curves:
        groups[row["problem"], row["q"], row["initial"], row["budget"], row["mode"]].append(row)
    for (problem, q, initial, total, mode), group in groups.items():
        fig, ax = plt.subplots(figsize=(8.1, 4.8), layout="constrained")
        for method in sorted({row["library"] for row in group}):
            records = [row for row in group if row["library"] == method]
            x = [row["evaluations"] for row in records]
            p25, med, p75 = np.asarray(
                [[row[field] for row in records] for field in ("p25", "median", "p75")],
                dtype=float,
            )
            (line,) = ax.plot(x, med, label=method)
            ax.fill_between(x, p25, p75, color=line.get_color(), alpha=0.13)
        ax.set(
            title=f"{problem}  |  mode {mode}, q={q}",
            xlabel="Completed evaluations (including initial design)",
            ylabel=group[0]["metric"].replace("_", " ").capitalize(),
        )
        ax.legend(ncol=2)
        ax.grid(alpha=0.20)
        fig.savefig(destination / f"{problem}-q{q}-n{initial}-b{total}-{mode}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    expectation = parser.add_mutually_exclusive_group()
    expectation.add_argument("--seeds", type=int, nargs="+", help="Expected seeds per cell.")
    expectation.add_argument("--manifest", type=Path, help="Campaign cell and seed manifest.")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for path in args.results
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = group_records(rows)
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = set(manifest["seeds"])
        scheduled = {
            (
                (cell["problem"], cell["q"], cell["initial"], cell["total"]),
                cell["mode"],
                cell["library"],
            )
            for cell in manifest["cells"]
        }
        if groups.keys() - scheduled or any(row["commit"] != manifest["commit"] for row in rows):
            raise ValueError("Results do not match the campaign manifest.")
        for key in scheduled:
            groups.setdefault(key, {})
    else:
        expected = set(args.seeds) if args.seeds else {row["seed"] for row in rows}
    if not expected or not groups:
        parser.error("Provide benchmark records or a manifest with cells and seeds.")
    summary, paired, curves = compare(groups, expected)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "paired.csv", paired)
    write_csv(args.output_dir / "curves.csv", curves)
    if args.plots:
        plot_curves(curves, args.output_dir)


if __name__ == "__main__":
    main()

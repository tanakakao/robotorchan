"""Minimal known-SCM benchmark mechanics and shared-initialization checks."""

from dataclasses import replace

import pytest
import torch

from robotorchan.benchmarks.causal import (
    make_causal_synthetic_scenario,
    run_causal_benchmark,
)
from robotorchan.causal import HardIntervention


def test_scm_has_known_joint_optimum_and_real_do_interventions():
    scenario = make_causal_synthetic_scenario()
    optimum = scenario.evaluate(HardIntervention({"X": (0.8 - 0.25) / 0.75, "Z": 0.65}))
    assert abs(optimum.values["Y"]) < 1e-12
    assert optimum.values["M"] == pytest.approx(0.8)
    assert scenario.evaluate(HardIntervention({"X": 0.2})).values["Y"] < 0
    with pytest.raises(ValueError, match="positive"):
        make_causal_synthetic_scenario(cost_joint=0.0)


def test_paired_initialization_metrics_and_one_real_gp_round():
    runs = run_causal_benchmark(
        seeds=(4,),
        n_initial_per_set=2,
        n_iterations=1,
        optimizer="sobol",
        num_restarts=2,
        raw_samples=16,
    )
    assert {item.method for item in runs} == {"cbo", "gp_bo", "random"}
    assert all(item.initial_points == 6 and len(item.steps) == 7 for item in runs)
    assert runs[0].steps[:6] == runs[1].steps[:6] == runs[2].steps[:6]
    for item in runs:
        assert item.steps[-1].evaluation == 7
        assert item.steps[-1].cumulative_cost > item.steps[5].cumulative_cost
        assert bool(torch.all(item.regret_curve >= 0))
        assert bool(torch.all(item.cost_curve[1:] > item.cost_curve[:-1]))
        assert bool(torch.all(item.best_curve[1:] >= item.best_curve[:-1]))
    assert set(runs[1].steps[-1].intervention_nodes) == {"X", "Z"}


def test_initial_seeds_repeat_and_validate_config():
    a = run_causal_benchmark(seeds=(11,), n_iterations=0)
    b = run_causal_benchmark(seeds=(11,), n_iterations=0)
    assert a == b
    with pytest.raises(ValueError, match="unique"):
        run_causal_benchmark(seeds=(1, 1), n_iterations=0)
    with pytest.raises(ValueError, match="at least 2"):
        run_causal_benchmark(n_initial_per_set=1, n_iterations=0)


def test_custom_scenario_full_intervention_is_independent_of_set_order():
    scenario = make_causal_synthetic_scenario()
    reordered = replace(
        scenario,
        intervention_sets=(("X", "Z"), ("Z",), ("X",)),
    )
    runs = run_causal_benchmark(
        seeds=(7,),
        scenario=reordered,
        n_initial_per_set=2,
        n_iterations=1,
        optimizer="sobol",
        num_restarts=2,
        raw_samples=16,
    )
    gp_bo = next(item for item in runs if item.method == "gp_bo")
    assert set(gp_bo.steps[-1].intervention_nodes) == {"X", "Z"}
    assert all(item.steps[:6] == runs[0].steps[:6] for item in runs)


def test_custom_scenario_rejects_missing_full_set_and_incomplete_costs():
    scenario = make_causal_synthetic_scenario()
    missing_full = replace(scenario, intervention_sets=(("X",), ("Z",)))
    with pytest.raises(ValueError, match="full-intervention"):
        run_causal_benchmark(scenario=missing_full, n_iterations=0)

    missing_cost = replace(scenario, costs={frozenset({"X"}): 1.0})
    with pytest.raises(ValueError, match="scenario.costs"):
        run_causal_benchmark(scenario=missing_cost, n_iterations=0)

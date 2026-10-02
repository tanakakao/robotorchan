# Notebook Coverage Gap Audit

## Scope

Phase 27 compares notebook metadata and executable workflows with the theory, runtime guides, and
current implementation contracts.

The goal is representative cross-layer coverage, not the Cartesian product of every model,
acquisition, optimizer, and search-space feature.

## Metadata corrections

The audit removes layer claims that were not explicitly demonstrated:

- `initialization` from notebooks 01, 02, 03, 04, 34, and 36;
- `loop` from notebook 28, which performs one active-learning update rather than an iterative loop;
- `fantasy` from notebook 33, where one-shot auxiliary variables and pending-point semantics are
  discussed but model fantasization is not demonstrated.

Notebook 33 keeps `initialization` because initialization is its primary executable contract.

## Material coverage gap: outcome-constrained BO

Known candidate constraints were already executable in notebook 31, and notebook 29 explained the
Objective / PosteriorTransform / outcome-constraint responsibility boundary. However, there was no
user-facing workflow that modeled an unknown constraint output and generated a candidate with
sample-wise feasibility in the acquisition.

Notebook 37 closes this gap with:

```text
objective GP + constraint GPs
        -> joint ModelListGP posterior
        -> posterior samples
        -> scalar MC objective
        + output constraint callables
        -> qLogExpectedImprovement
        -> candidate optimization
        -> objective + constraint evaluation
```

The notebook intentionally keeps the optimizer bounds broad. The unknown feasible interval is not
converted into a known candidate constraint.

## Cross-layer combinations not duplicated

The following combinations are useful but do not justify dedicated notebooks in the current
representative suite:

- Mixed x Multi-objective;
- Mixed x candidate constraints;
- MultiTask x Multi-objective;
- High-dimensional x TuRBO.

Their component contracts are already covered independently, and runtime tests should carry
combinatorial compatibility where needed. A dedicated notebook should be added only when the
combination introduces a distinct user-facing semantic contract.

Examples:

- mixed categorical search changes acquisition optimization geometry, but does not redefine Pareto
  dominance;
- MultiTask output structure and multi-objective utility are separate concerns;
- TuRBO localizes the public search region while high-dimensional surrogate reduction remains a
  model-side concern.

## Robust coverage

Robust observation modeling and robust input-perturbation BO remain separate notebooks
intentionally. Combining them would suggest that one robustness layer requires the other.

## Constraint coverage after Phase 27

The executable suite now distinguishes:

- known candidate constraints: notebook 31;
- unknown output / black-box constraints: notebook 37;
- Objective / PosteriorTransform responsibility: notebook 29;
- multi-objective BO: notebook 27;
- robust input perturbation: notebook 35.

This separation mirrors the runtime ownership boundaries instead of hiding them behind one generic
"constraints" example.

## Remaining extensions

The following are future extensions rather than current documentation gaps:

- constrained multi-objective BO;
- Hypervolume Knowledge Gradient integration;
- ensemble-aware robotorchan-specific active learning;
- classification active learning once classification models enter scope;
- safe BO with explicit safety guarantees.

These should receive notebooks only after their runtime contracts are intentionally supported and
integration-tested.

# Model × acquisition × optimizer E2E validation

Optimizer E2E validation confirms that backends operate with real BoTorch model
posteriors and acquisition functions, rather than only synthetic acquisition
modules.

The E2E matrix intentionally follows optimizer capabilities instead of forcing
every model/acquisition/backend combination. Executable optimizer E2E coverage is owned by
`tests/optim/integration/`.

## Continuous scalar path

The CI smoke matrix fits a `SingleTaskGP` and exercises:

- analytic Expected Improvement with the BoTorch-native optimizer,
- analytic UCB with Sobol sampling,
- analytic UCB with the native genetic algorithm,
- analytic UCB with particle swarm optimization.

These tests validate posterior evaluation, acquisition evaluation, candidate
shape, dtype preservation, and bounds.

## Constraint regressions

Regression coverage preserves two important constraint-handling contracts:

- mixed GA now returns the raw acquisition value, not its penalized ranking score;
- CMA-ES tracks the best candidate by its constrained/penalized objective rather
  than by unconstrained acquisition value.

CMA-ES regression coverage is conditional on the optional `cmaes` dependency.

## Scope boundaries

The E2E contract does not assert that one optimizer is universally superior.
Runtime and candidate-quality comparisons belong to the benchmark harness.

Vector-valued NSGA-II remains a separate contract because qEHVI/qNEHVI are
scalar acquisition functions even though their underlying BO problem is
multi-objective. Mixed-variable E2E and broader model families are capability
tests and can be expanded without changing the optimizer API.

# NSGA-II and Bayesian optimization

Phase 12 evaluated whether NSGA-II should be another backend for
\`optimize_acqf\`. The answer is no for ordinary BoTorch multi-objective
acquisition optimization.

qEHVI, qNEHVI, qLogEHVI, qLogNEHVI and similar BoTorch acquisition functions
already reduce multi-objective improvement to a scalar acquisition value.
That scalar can be optimized by the existing BoTorch, DE, CMA-ES, GA, or
hybrid backends. Running NSGA-II on that scalar does not add multi-objective
semantics.

NSGA-II is useful when the numerical optimization target itself is vector
valued. Examples include direct optimization of multiple surrogate posterior
statistics, explicit multi-criterion acquisition vectors, or hybrid BO /
evolutionary workflows that intentionally preserve a Pareto set before a
later selection step.

For that reason robotorchan exposes a separate API:

\`\`\`python
from robotorchan.optim.backends import optimize_vector_nsga2

pareto_X, pareto_values = optimize_vector_nsga2(
    objective=vector_objective,
    bounds=bounds,
    population_size=128,
    generations=100,
    seed=123,
)
\`\`\`

The objective receives a batched \`[population, d]\` tensor and returns
\`[population, m]\`. Every objective is maximized. The backend returns only
the non-dominated front from the final population.

The implementation is native PyTorch and uses batched objective evaluation,
non-dominated sorting, crowding distance, binary tournament selection,
crossover, mutation, and elitist environmental selection.

Current scope is continuous variables without CandidateConstraints. It is
deliberately not wired into scalar \`optimize_acqf\`. Constraint-aware
multi-objective evolutionary optimization and NSGA-III are evaluated in
later phases rather than being conflated with BoTorch's scalar acquisition
contract.

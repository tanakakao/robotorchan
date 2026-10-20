# Candidate constraints across optimizer backends

Candidate-space constraints are specified with the shared
`CandidateConstraints` contract. They restrict which input candidates
may be proposed; they are distinct from outcome constraints used to
construct constrained acquisition functions.

## Backend-specific enforcement

| Backend | Candidate constraint handling |
| --- | --- |
| BoTorch / BoTorch-mixed | Forward supported linear and nonlinear constraints to BoTorch |
| Differential Evolution | Wrap shared violation evaluation in SciPy `NonlinearConstraint` |
| CMA-ES | Penalize constraint violation in the search objective |
| Continuous GA | Rank feasible candidates ahead of infeasible candidates |
| Mixed-variable GA | Use feasibility-first ranking with mixed-domain operators |
| PSO | Penalize violation in particle scores; return the raw acquisition value |
| Hybrid | Pass constraints to both global and BoTorch local stages; handling depends on the global backend |
| Random / Sobol | Candidate constraints are unsupported and explicitly rejected |
| PyTorch candidate generator | Candidate constraints are unsupported and explicitly rejected |

The derivative-free backends reuse the same constraint evaluation
contract but **do not share a single penalty algorithm**.
CMA-ES and PSO accept a positive `constraint_penalty`; do not assume this
parameter controls Differential Evolution or GA feasibility handling.
Hybrid uses the selected global backend's handling (DE, CMA-ES, or Mixed GA)
followed by BoTorch local refinement; its stages do not share one penalty rule.

The shared `equality_tolerance` defaults to `1e-6`. Supported
constraints include intra-point and inter-point q-batch conditions.
Derivative-free backends require valid tensor outputs from nonlinear
constraint callables, not differentiability. Gradient-based BoTorch
optimization has additional differentiability requirements.

## Feasibility and validation

Constraint handling does not guarantee that every optimization run
returns a feasible candidate. In particular, penalty-based CMA-ES or PSO
runs can return infeasible results, and feasibility-first ranking
depends on the evaluated population. Check the returned candidate
against the original constraints when strict feasibility matters.

Do not silently replace an unsupported candidate constraint with an
outcome feasibility weight or an approximate search-space penalty.
Inspect the selected backend's capability contract before dispatch.

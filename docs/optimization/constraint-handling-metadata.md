# Constraint-handling capability metadata

Optimizer support for a constraint is not a boolean property alone. A backend
can accept the same `CandidateConstraints` contract while enforcing it with
different numerical semantics.

`OptimizerCapabilities.constraint_handling` therefore records one mode per
constraint class:

- `native`: delegated to the optimizer's native constrained optimization;
- `penalty`: total violation augments the search objective;
- `feasibility_first`: feasible candidates dominate infeasible candidates;
- `repair`: invalid candidates are transformed into feasible candidates;
- `projection`: candidates are projected onto a feasible set;
- `rejection`: invalid candidates are discarded/resampled;
- `unsupported`: the backend must reject the request before optimization.

The enum deliberately includes modes that are not yet used. Metadata describes
implemented behavior only; declaring a mode does not itself implement it.

## Current mapping

| Backend family | Linear inequality | Linear equality | Nonlinear | Inter-point nonlinear |
| --- | --- | --- | --- | --- |
| BoTorch | native | native | native | native |
| BoTorch mixed | native | native | native | unsupported |
| DE | feasibility_first | feasibility_first | feasibility_first | feasibility_first |
| CMA-ES | penalty | penalty | penalty | penalty |
| GA | feasibility_first | feasibility_first | feasibility_first | feasibility_first |
| Mixed GA | feasibility_first | feasibility_first | feasibility_first | feasibility_first |
| PSO | penalty | penalty | penalty | penalty |
| Hybrid | penalty | penalty | penalty | penalty |
| Torch / Random / Sobol / NSGA-II | unsupported | unsupported | unsupported | unsupported |

The existing boolean fields remain the coarse compatibility surface used by
runtime validation. The handling metadata is the precise semantic contract and
will allow later phases to compare penalty handling with Deb-style
`feasibility_first` without pretending they are equivalent.

`CandidateConstraints` remains the only candidate-constraint representation.
This metadata does not introduce a second constraint DSL.

# Multi-objective benchmark suite

The suite now includes:

| Problem | Variables | Objectives | Directions | Pareto metadata |
| --- | ---: | ---: | --- | --- |
| Biobjective linear | 1 | 2 | maximize/maximize | sampled analytic front |
| Branin-Currin | 2 | 2 | maximize/maximize | reference point only |
| DTLZ2 | 7 | 3 | minimize/minimize/minimize | sampled analytic front |
| ZDT1 | 6 | 2 | minimize/minimize | sampled analytic front |

Branin-Currin uses the standard negative-Branin/Currin convention,
with normalized inputs on [0, 1]^2 and a conservative maximization
reference point (-310, -1). Its true Pareto front is not claimed to
be analytically known; `reference_front` is intentionally absent.
At the Currin boundary x2=0, the exponential term is defined by its
continuous limit, avoiding NaNs.

`register_multiobjective_problems(registry)` registers all four
problems. Hypervolume comparisons should use a common fixed reference
point, and transform minimization objectives consistently before
computing hypervolume. This phase extends benchmark problem coverage,
not the multiobjective BO acquisition strategy.

# Multi-objective heterogeneous benchmarks

These problems model **Strength** and **Conductivity** as two continuous
regression objectives to maximize, with binary outcome constraints:

- `strength_conductivity_pass_tradeoff`: one Pass/Fail constraint.
- `strength_conductivity_two_pass`: two Pass/Fail constraints.

Both use `x0,x1 in [0,1]`. The objective functions are

```text
Strength     = 1 - (x0 - 0.8)^2 - 0.25*(x1 - 0.3)^2
Conductivity = 1 - (x0 - 0.2)^2 - 0.25*(x1 - 0.7)^2
```

Feasibility truth is defined by signed margins `g(X) >= 0`.
The first margin is `x1 - 0.25 - 0.5*x0`; the second margin
is `0.8 - (x0-0.5)^2 - (x1-0.5)^2`.

Use `multiobjective_pass_labels(X)` or
`multiobjective_two_labels(X)` for canonical 0/1 classification
targets. **Do not train binary classifiers on signed truth margins.**

The benchmark runner records truth objectives and constraint margins.
`hypervolume_curve` measures the feasible dominated hypervolume
relative to the reference point `(0, 0)`. These are truth-based
evaluation metrics, not acquisition functions.

The phase supplies problem definitions, labels, reproducibility tests
and feasible hypervolume evaluation. Training heterogeneous surrogate
models and optimizing constrained qEHVI/qNEHVI are separate
end-to-end integration checks.

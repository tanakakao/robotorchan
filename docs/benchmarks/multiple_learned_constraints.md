# Multiple learned outcome constraints

Phase 19 adds two single-objective benchmarks maximizing Yield:

- `yield_two_binary_constraints`: two binary Pass/Fail conditions.
- `yield_binary_continuous_constraints`: one binary Pass/Fail
  condition and one continuous temperature limit.

The objective is `1-(x0-0.8)^2-(x1-0.2)^2` for `x0,x1 in [0,1]`.
All truth constraints follow `g(X) >= 0`.

| Constraint | Truth margin | Interpretation |
| --- | --- | --- |
| Pass | `x1 - 0.2 - 0.4*x0` | Binary feasibility |
| Reliability | `0.4 - x0` | Second binary feasibility |
| Temperature | `0.8 - x1` | Continuous limit |

The first problem uses Pass and Reliability. The second uses Pass and
Temperature. The constrained optima are respectively
`(0.4, 0.36)` with Yield `0.8144`, and
`(0.8/1.16, 0.2+0.4*(0.8/1.16))` with Yield approximately
`0.911724`.

`binary_constraint_labels(X)` produces canonical 0/1 labels
for the first problem's two binary constraints. In the mixed problem,
train the binary Pass model using the **first** label column and a
regression model on the continuous Temperature margin.

Signed truth margins support reproducible feasibility metrics; they
are not classifier training labels. The phase provides benchmark
fixtures, registry exports and tests, but does not yet train surrogate
models or evaluate joint predictive feasibility.

# Extended benchmark metrics

Phase 21 adds metrics that complement the existing simple-regret,
hypervolume and feasibility curves.

| Metric | API | Direction |
| --- | --- | --- |
| Trapezoidal area under curve | `area_under_curve(curve, budgets)` | Depends on curve |
| Best feasible objective so far | `best_feasible_value_curve(trajectory, maximize=True)` | Higher for maximize |
| First jointly feasible evaluation | `first_feasible_evaluation(constraints)` | Lower |
| Binary negative log-likelihood | `log_loss(labels, probabilities)` | Lower |
| Positive-class expected calibration error | `probability_calibration_error(labels, probabilities)` | Lower |

The area-under-curve function accepts evaluation or cumulative-cost
budgets; budgets must increase strictly. Infinite curve values propagate
to the integral, so use a defined policy for runs with no feasible
candidate before comparing areas.

The best-feasible-value curve returns NaN before the first feasible
point. First feasibility uses a **one-based index including initial
design evaluations**, and returns `None` if no feasible point was
observed.

Log loss clamps predicted probabilities only for numerical stability.
Calibration error partitions **positive-class probabilities** into
equal-width bins on `[0, 1]`, with 1 included in the last bin.
This is not confidence-based multiclass ECE.

Use the same fixed reference set, evaluation budget and prediction
schedule for fair active-learning comparisons. These functions do not
perform surrogate fitting or acquire new observations.

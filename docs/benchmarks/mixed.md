# Mixed-variable benchmark problems

The benchmark suite represents categorical variables as **integer-coded
nominal labels**, not as ordinal numeric distances. Candidate generators
must return valid integer codes within bounds. No one-hot encoding is
required by the problem contract.

| Problem | Inputs | Objective | Constraints |
| --- | --- | --- | --- |
| mixed_quadratic | continuous, integer, category | minimize; optimum 0 | none |
| categorical_switch | continuous, category | maximize; optimum 1 | none |
| mixed_process_yield | temperature, integer cycle count, recipe category | maximize; optimum 1 | recipe-dependent temperature limit; cycles <= 4 |
| mixed_category_interaction | continuous, two categorical choices | minimize; optimum 0 | none |

**Process yield** has recipe-dependent response centers and upper
temperature limits. The feasible optimum is (0.7, 3, 1).
This exercises the interaction between categorical choices and
continuous outcome constraints.

**Category interaction** has a non-additive lookup table for the
continuous response center and categorical penalty. Its optimum
is (0.85, 1, 1), demonstrating a case where two nominal variables
jointly determine the landscape.

The existing `sobol_initial_design` and `random_candidates` functions
round discrete coordinates. For more sophisticated mixed-variable
optimization, use a compatible candidate generator with the same
`(q, d)` tensor contract.

Register all problems using `register_mixed_problems(registry)`.

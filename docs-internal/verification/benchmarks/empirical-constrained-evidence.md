# Phase 29: constrained empirical eligibility and evidence

Phase 29 audits eligibility rather than silently unblocking constrained GP
acquisitions. The two benchmark families are `strength_pass` (one regression
objective and binary Pass label) and `strength_conductivity_pass` (two regression
objectives and binary Pass label).

| Problem | Random / Sobol | GP acquisitions |
| --- | --- | --- |
| strength_pass | Eligible baseline | qEI / qNEI blocked |
| strength_conductivity_pass | Eligible baseline | qEHVI / qNEHVI blocked |

The regression objectives are observed outcomes. The binary Pass label is a
separate training target. The signed feasibility margin belongs to the
evaluation oracle and **must not** be supplied to the candidate strategy.
Constrained GP methods cannot be certified until the strategy receives
observed 0/1 labels, fits an appropriate classifier, exposes predictive Pass
probabilities, and composes a probability-based constraint with the acquisition
function. An unconstrained acquisition must never be presented as a constrained
one.

`constrained_evidence.py` makes this distinction executable: it reports
explicit eligibility, generates canonical observed labels, checks evaluation
margins against binary feasibility, and reports feasible counts/rates and
zero-feasible status for complete baseline trajectories.

**Acceptance:** blocked methods remain blocked; no GP acquisition is claimed
eligible merely because a signed deterministic margin exists. No measured
constrained performance result is claimed in this phase. Full baseline
cohort execution and statistical reporting remain separate work.

# Phase 12: q-batch compatibility and closure

This document records the implemented q>1 acquisition paths without treating
registry declarations as proof of every model/optimizer combination.

## Acquisition semantics

| Family | Registered acquisitions | q>1 behavior |
| --- | --- | --- |
| BoTorch native BO | qLogEI, qLogNEI, qPI, qUCB, qSimpleRegret, qLogEHVI, qLogNEHVI, qLogNParEGO | Native BoTorch batch objectives |
| BoTorch lookahead | qKnowledgeGradient, qMultiFidelityKnowledgeGradient | One-shot; specialized optimizer path |
| Joint Gaussian AL | qJointPosteriorEntropy, qJointExpectedPredictiveInformationGain | Joint posterior covariance |
| Conditional joint Gaussian AL | qConditionalExpectedPredictiveInformationGain | Joint Gaussian conditional on optional pending locations; not fantasization |
| Joint binary AL | qBALD | Joint binary labels; maximum q=8 |
| Marginal aggregate AL | qPosteriorVariance, qPosteriorStd, qStraddle, qRandomizedStraddle, qBoundaryVariance, qExpectedPredictiveInformationGain | Aggregates pointwise scores; no batch-diversity guarantee |
| Marginal classification AL | qPredictiveEntropy, qMarginUncertainty, qProbabilityVariance, qLatentStraddle | Aggregates pointwise scores; no batch-diversity guarantee |
| Single-candidate AL | ExpectedPredictiveInformationGain, BALD, PredictiveEntropy, MarginUncertainty, ProbabilityVariance, LatentStraddle | q=1 only; use corresponding q variant where available |

Names in the table use common abbreviations where unambiguous. The authoritative
public names and `max_q` limits are in
`src/robotorchan/acquisition/registry.py`.

## What is directly verified

- `qLogExpectedImprovement`, `qProbabilityOfImprovement`,
  `qSimpleRegret`, and `qUpperConfidenceBound`: `SingleTaskGP`,
  continuous unconstrained optimization, q=2 and q=4, with BoTorch and
  Sobol backends. Evidence:
  `tests/optim/integration/test_phase9_native_joint_q_batch.py`.
- Joint Gaussian qEPIG: q=2 candidate optimization with fixed features.
  Evidence: `tests/acquisition/test_q_joint_epig_e2e.py`.
- Conditional qEPIG: q=2 candidate optimization with a pending location.
  Evidence: `tests/acquisition/test_q_conditional_epig_async_e2e.py`.
- Joint qEPIG mixed and constrained workflows have dedicated E2E tests:
  `tests/acquisition/test_q_joint_epig_mixed_e2e.py` and
  `tests/acquisition/test_q_joint_epig_constraints_e2e.py`.

These are examples of scoped evidence, not a universal compatibility guarantee.
The audit report records declared capabilities separately from runtime evidence.

## Optimizer and variable-type boundaries

- Joint q optimization is not interchangeable with sequential candidate
  generation. In particular, mixed BoTorch q-batch uses a sequential path.
- Mixed, integer, categorical, fixed-feature, and constrained support depends
  on the chosen optimizer backend and acquisition; a scalar q>1 objective
  alone does not guarantee those combinations.
- One-shot KG and multi-fidelity KG need their specialized optimization
  contracts. Generic q-batch optimizer evidence does not validate them.
- Vector-objective NSGA-II and scalar joint-q acquisition optimization solve
  different problems; do not conflate them.
- Pending locations in conditional qEPIG do not imply posterior fantasization
  over uncertain pending outcomes.

## Completion assessment

The planned q>1 acquisition families and core optimizer contracts are
implemented, and the named integration paths have runtime evidence. Phase 12
can close as a **scoped q-batch implementation milestone**, not as a claim
that every acquisition works with every model, optimizer, variable type, and
constraint. No further test-matrix expansion is required for this milestone.

For future changes, add an implementation and a focused runtime test only
when a concrete missing combination is requested or a reproducible failure
is found. Avoid extending the audit solely to increase pair counts.

# q-batch Phase 1: static implementation inventory

This report audits the `main` branch as inspected on 2026-11-05. It is a **static source audit**, not evidence of successful q=2 or q=4 runtime tests. Do not equate capability declarations with verified behavior.

## Status vocabulary

- **STRUCTURAL**: source passes q through or reshapes a joint q-by-d batch.
- **DECLARED**: capability metadata claims q support, without execution evidence.
- **LIMITED_Q1**: source explicitly rejects q>1.
- **SEPARATE**: different problem/API, not a scalar joint-acquisition optimizer.
- **UNVERIFIED**: runtime and numerical behavior have not been tested.

## Scalar optimizer dispatch inventory

All 11 names below are accepted by `optim/dispatch.py::optimize_acqf`. The capabilities in `optim/capabilities.py` declare `q_batch=True` for all of them.

| Dispatch name | Implementation | q>1 static evidence | Runtime |
| --- | --- | --- | --- |
| botorch | backends/botorch.py::optimize_acqf_botorch | passes q to BoTorch; sequential option | UNVERIFIED |
| torch_adam | backends/torch.py::optimize_acqf_torch | passes q to common Torch optimizer | UNVERIFIED |
| torch_adamw | backends/torch.py::optimize_acqf_torch | passes q to common Torch optimizer | UNVERIFIED |
| torch_sgd | backends/torch.py::optimize_acqf_torch | passes q to common Torch optimizer | UNVERIFIED |
| random | backends/sampling.py::optimize_acqf_sampling | samples [num_samples,q,d] | UNVERIFIED |
| sobol | backends/sampling.py::optimize_acqf_sampling | Sobol dimension q*d, reshapes to [num_samples,q,d] | UNVERIFIED |
| de | backends/differential_evolution.py::optimize_acqf_de | flattens q*d and reshapes [q,d] | UNVERIFIED |
| cmaes | backends/cmaes.py::optimize_acqf_cmaes | reshapes candidates [1,q,d] | UNVERIFIED |
| ga | backends/genetic_algorithm.py::optimize_acqf_ga | population reshaped [population,q,d] | UNVERIFIED |
| pso | backends/pso.py::optimize_acqf_pso | swarm positions q*d, reshaped [swarm,q,d] | UNVERIFIED |
| hybrid | backends/hybrid.py::optimize_acqf_hybrid | forwards q to inner optimizers | UNVERIFIED |

Mixed dispatch paths: `optimize_acqf_mixed_botorch`, `optimize_acqf_mixed_ga`, and `optimize_acqf_mixed_cmaes` are selected conditionally from the corresponding dispatch names. Mixed GA expands integer, categorical, and fixed dimensions for q rows. Mixed CMA-ES encodes q*d variables. All are STRUCTURAL / UNVERIFIED. `optimize_vector_nsga2` and `optimize_vector_nsga2_mixed` solve vector-objective Pareto optimization, not scalar joint-acquisition optimization; both have `q_batch=False` capabilities (SEPARATE).

Sequential dispatch is exposed only for continuous BoTorch in the named dispatcher; mixed BoTorch explicitly rejects sequential mode. Derivative-free dispatch rejects sequential mode and directs callers to `optimize_acqf_sequential`. Do not infer that all optimizers support sequential batches.

## Acquisition registry: all 21 registered names

`acquisition/registry.py` registers 11 active-learning functions with `max_q=1` (LIMITED_Q1):

- BALD
- BoundaryVariance
- ExpectedPredictiveInformationGain
- LatentStraddle
- MarginUncertainty
- PosteriorStd
- PosteriorVariance
- PredictiveEntropy
- ProbabilityVariance
- RandomizedStraddle
- Straddle

`active_learning/classification.py::_require_q_one` enforces the restriction for classification AL; `active_learning/epig.py`, `variance.py`, `straddle.py`, and `randomized_straddle.py` include q=1 restrictions. `PosteriorStd` inherits `PosteriorVariance`'s restriction.

The following 10 BoTorch-native registry entries declare `max_q=None` (DECLARED / UNVERIFIED):

- qLogExpectedImprovement
- qLogNoisyExpectedImprovement
- qUpperConfidenceBound
- qProbabilityOfImprovement
- qSimpleRegret
- qKnowledgeGradient
- qMultiFidelityKnowledgeGradient
- qLogExpectedHypervolumeImprovement
- qLogNoisyExpectedHypervolumeImprovement
- qLogNParEGO

The registry is **not** an exhaustive inventory of all BoTorch acquisitions usable by callers. Do not infer unsupported status for unregistered BoTorch acquisitions.

## Public acquisitions and composition outside the registry

| Source | Public callable(s) | q>1 status |
| --- | --- | --- |
| acquisition/classification_constraints.py | ClassificationProbabilityAcquisition | LIMITED_Q1 |
| acquisition/classification_constraints.py | RobustClassificationProbabilityOfFeasibility, FeasibilityWeightedAcquisition, IndependentFeasibilityAggregator | q-aware aggregation path; UNVERIFIED |
| acquisition/composition.py | make_qei_acquisition, make_qnei_acquisition, make_qehvi_acquisition, make_qnehvi_acquisition, make_qlognparego_acquisition | construct q acquisition; UNVERIFIED |
| acquisition/composition.py | make_continuous_constrained_qei_acquisition, make_continuous_constrained_qnei_acquisition, make_mixed_constrained_qei_acquisition, make_multiple_learned_constrained_qnei_acquisition, make_constrained_qehvi_acquisition | q composition paths; UNVERIFIED |
| acquisition/composition.py | make_deterministic_pof_acquisition, make_classification_feasibility_bridge, make_continuous_constraint_bridge, make_sample_classification_feasibility_bridge, make_probability_objective_bridge | adapters / bridges; assess within composed acquisition |
| acquisition/non_gp.py | make_non_gp_acquisition, validate_non_gp_acquisition | model-dependent; UNVERIFIED |
| acquisition/sampling/thompson.py | select_thompson_candidates | candidate selection, not a scalar joint-acquisition function; SEPARATE |

Other composition helpers, objectives and bridge classes are not independent acquisition functions; assess their q shape contracts as dependencies of the constructors above.

## Cross-product matrix interpretation

For each registered acquisition and named optimizer, the static q>1 matrix is:

- All **11 LIMITED_Q1** acquisitions × all **11** dispatch names: **blocked by acquisition q=1 semantics** until a mathematically defined batch extension is implemented. No claim that every dispatcher directly checks the acquisition registry.
- All **10 DECLARED** q acquisitions × all **11** dispatch names: **110 potentially relevant combinations, all runtime UNVERIFIED**. This does **not** mean 110 supported combinations. One-shot, posterior, differentiability, constraints, model type, and mixed semantics may eliminate combinations.
- Registry-external constructors and adapters require separate instance-based runtime tests; they cannot be counted as 1:1 independent acquisitions.
- NSGA-II and mixed NSGA-II are excluded from the scalar joint-acquisition cross-product.

## Follow-up verification plan

1. Add executable smoke tests for q=1,2,4 across all 11 dispatch names using a differentiable joint q acquisition; verify [q,d] candidates and scalar score.
2. Exercise mixed backends with integer/categorical domains, fixed features and feasible constraints. Distinguish intra-point and inter-point constraints.
3. Test q=2 and q=4 for each registered BoTorch-native acquisition using compatible model, objective, sampler and one-shot initialization. Record explicit incompatibilities instead of silently falling back.
4. For each q=1 AL acquisition, keep the explicit rejection until a documented joint or sequential mathematical definition exists; do not implement sum-of-single-point scores under a native q acquisition name.
5. Verify pending points, sequential behavior, reproducibility, gradient and finite-value contracts.
6. Extend the existing `scripts/audit_acquisition_optimizer_matrix.py` report with q and verification status rather than introducing a competing registry.

**Phase 1 conclusion:** static inventory complete for the 11 scalar dispatcher names, 21 registered acquisition names, and identified public acquisition composition/adapter paths. End-to-end and numerical verification remains open; Phase 1 should not be marked fully verified.

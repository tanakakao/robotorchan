# Runtime end-to-end validation matrix

This document records representative executable workflows. It is intentionally narrower than
the model registry: structural capability does not imply that every model/acquisition/optimizer
combination has an end-to-end test.

| Workflow | Posterior | Sampler / reduction | Acquisition | Candidate generation | Status |
| --- | --- | --- | --- | --- | --- |
| Continuous BO | Gaussian | Sobol QMC | qLogEI | optimize_acqf | Runtime validated |
| Mixed BO | Gaussian | Sobol QMC | qLogEI | optimize_acqf_mixed | Runtime validated |
| MultiTask / multi-output | Gaussian multi-output | scalarization / Sobol QMC | qLogEI | acquisition runtime | Runtime validated |
| Multi-objective | Gaussian multi-output | Sobol QMC | qLogEHVI / qLogNEHVI | acquisition runtime | Runtime validated |
| Multi-fidelity | Gaussian | Sobol QMC | qMFKG + cost + projection | continuous optimization path | Runtime validated |
| Mixed multi-fidelity | Gaussian | Sobol QMC | qMFKG + cost + projection | mixed one-shot optimizer | Runtime validated |
| High-dimensional reduction | Gaussian | Sobol QMC | qLogEI / qKG | representative runtime | Runtime validated |
| Mixed high-dimensional | Gaussian | Sobol QMC | qLogEI / qKG | mixed one-shot KG path | Runtime validated |
| Empirical non-GP ensemble | EnsemblePosterior | IndexSampler | qLogEI | acquisition runtime | Runtime validated |
| Knowledge Gradient | Gaussian fantasy model | Sobol QMC | qKG | mixed one-shot path where needed | Runtime validated |
| Robust models | family-specific | compatible BoTorch path | representative MC acquisition | family tests | Partially validated |
| Active learning | marginal / joint Gaussian | acquisition-owned semantics | local AL acquisitions | q=1 evaluation | Runtime validated |

## Interpretation

Runtime validated means that at least one representative executable path is covered. It does not
mean every registered model in the family has been benchmarked with every compatible acquisition.

Candidate generation is required for the core continuous BO, mixed BO, mixed qKG, and mixed qMFKG
workflows because optimizer semantics are part of those public contracts. Multi-objective,
robust, and active-learning coverage remains acquisition/runtime focused unless candidate
optimization adds distinct semantics.

## Remaining validation gaps

The remaining gaps are validation-depth items rather than known correctness defects:

- robust MultiTask and Mixed robust families are not exhaustively exercised through optimizers;
- multi-output acquisition coverage is representative rather than exhaustive across all models;
- empirical ensemble active-learning reduction semantics are intentionally unsupported;
- multi-fidelity high-dimensional and robust public surfaces now have representative runtime
  evidence; specialized research-gated combinations remain intentionally excluded;
- q > 1 mixed one-shot optimization remains intentionally unsupported.

These gaps must not be converted into capability claims without executable tests.


## Phase 40 scope refresh

The multi-fidelity cross-capability statement above is updated from the historical audit state.
Public high-dimensional MF models now include deterministic reductions and MAP-SAAS variants, and
public robust MF models include replicate-noise and heteroskedastic variants. These families have
representative native MF-KG evidence.

The remaining validation-depth priority is therefore robust / high-dimensional MultiTask and
Kronecker acquisition coverage. Classification and ordinal models are not part of this runtime
program and will be developed separately.


## Phase 13 acquisition compatibility closure

Phase 13 cross-checks the recently strengthened regression families against registry-driven
acquisition and sampler selection rather than relying only on family-local runtime tests.

Executable compatibility assertions now cover:

- replicate-noise and heteroskedastic multi-fidelity models with
  `qMultiFidelityKnowledgeGradient`;
- Gaussian sampler selection for those robust multi-fidelity posteriors;
- NGBoost with sample-based `qLogExpectedImprovement` and a Gaussian-distribution sampler;
- explicit rejection of fantasy-dependent `qKnowledgeGradient` for NGBoost.

This separates two meanings that should not be conflated: a model may provide a sampleable
predictive distribution while still lacking the fantasy-model lifecycle required by KG-style
lookahead acquisitions.

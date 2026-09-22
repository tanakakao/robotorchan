# Acquisition integration status

The acquisition layer follows a BoTorch-first policy: native BoTorch acquisitions are used
directly, while robotorchan owns only functionality that adds a distinct contract or algorithm.

| Area | Preferred path |
| --- | --- |
| Standard BO | Native LogEI, LogPI, UCB, qLogEI, qLogNEI, qUCB |
| Posterior sampling | `select_thompson_candidates` |
| Regression AL | `PosteriorVariance`, `PosteriorStd`; native qNIPV for batch AL |
| Level-set learning | `Straddle`, `BoundaryVariance`, `RandomizedStraddle` |
| Information-theoretic BO | Native MES / GIBBON |
| Lookahead | Native qKG / qMultiStepLookahead |
| Multi-objective | Native qLogEHVI / qLogNEHVI / qLogNParEGO; HVKG is not yet integration-tested |
| Multi-fidelity | Native MF-KG and cost-aware utilities |
| Predictive AL | `ExpectedPredictiveInformationGain` |

## Explicit limitations

Robotorchan-specific q=1 regression acquisitions do not claim structured-output or arbitrary
batch support. EPIG currently accepts an unbatched finite target set and rejects ensemble
posteriors explicitly. `PosteriorVariance`, `PosteriorStd`, `Straddle`, `BoundaryVariance`, and
`RandomizedStraddle` also reject ensemble posteriors until reduction semantics are defined.
Thompson sampling requires an explicit objective for multi-output models.
`BoundaryVariance` is a robotorchan-specific heuristic rather than a named literature method.

These restrictions are deliberate: unsupported posterior shapes should fail clearly rather than
silently reducing the wrong dimension.

## Final integration rule

New acquisition functionality should first check whether BoTorch already exposes the required
algorithm. Native functionality should be integration-tested and documented instead of wrapped.
A robotorchan implementation is justified when it supplies a genuinely missing algorithm or a
robotorchan-specific contract.


## Audit follow-ups

The current acquisition milestone is feature-complete for the scoped single-objective BO,
regression active learning, level-set learning, information-theoretic BO, lookahead,
multi-objective BO, and multi-fidelity/cost-aware integration.

Useful future extensions are intentionally left as separate work rather than implied support:

- integration coverage for Hypervolume Knowledge Gradient in multi-objective BO;
- constrained/custom-objective multi-objective acquisition examples;
- explicit ensemble reduction semantics for robotorchan-specific regression acquisitions;
- optional multi-fidelity MES coverage where it provides value beyond MF-KG.

These are extension items, not compatibility gaps in the documented current contracts.


## Theory-to-implementation correspondence

The theory hierarchy intentionally covers a broader decision space than robotorchan's local
acquisition package. Use the following ownership categories when reading the documentation.

| Theory / decision area | Runtime owner | robotorchan correspondence |
| --- | --- | --- |
| Improvement / confidence-bound BO | BoTorch native | integration-tested; no local wrapper |
| Batch / noisy BO | BoTorch native | integration-tested; no local wrapper |
| Information-theoretic BO | BoTorch native | MES / GIBBON integration path |
| Lookahead | BoTorch native | qKG / qMultiStepLookahead integration path |
| Multi-objective BO | BoTorch native | qLogEHVI / qLogNEHVI / qLogNParEGO integration path |
| Constraints | BoTorch composition | theory coverage does not imply a robotorchan acquisition class |
| Regression uncertainty AL | robotorchan + BoTorch | `PosteriorVariance`, `PosteriorStd`; native qNIPV for batch AL |
| Predictive AL | robotorchan | `ExpectedPredictiveInformationGain` |
| Level-set learning | robotorchan | `Straddle`, `RandomizedStraddle`, `BoundaryVariance` |
| Multi-fidelity / cost-aware BO | BoTorch native | MF-KG and cost-aware integration path |
| Posterior sampling | robotorchan utility over BoTorch | `select_thompson_candidates` |
| Non-GP acquisition compatibility | robotorchan contract helper | `make_non_gp_acquisition`, `validate_non_gp_acquisition` |

The local public API is therefore intentionally smaller than the theory catalog. A theory chapter
is not a request to duplicate a BoTorch acquisition under a robotorchan alias.

### Public API contract

The locally owned acquisition surface is:

```text
BoundaryVariance
ExpectedPredictiveInformationGain
PosteriorStd
PosteriorVariance
RandomizedStraddle
Straddle
make_non_gp_acquisition
select_thompson_candidates
validate_non_gp_acquisition
```

`tests/acquisition/test_public_api.py` locks this list explicitly. Additions require a distinct
robotorchan algorithm or contract; documentation coverage alone is not sufficient justification.

### Correspondence audit result

The theory chapters, practical optimization guides, status table, package exports, and public-API
test are consistent with the current ownership policy. Remaining items listed under
**Audit follow-ups** are intentionally unsupported or not yet integration-tested rather than
silently implied by the theory documentation.

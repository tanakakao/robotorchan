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

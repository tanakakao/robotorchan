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

Mixed qKG and qMFKG support covers model construction, fantasy-aware acquisition evaluation,
cross-category fantasy decision points, and correctness-first candidate generation for `q=1`.
`optimize_mixed_one_shot_acqf` uses exact row-wise categorical enumeration so fantasy rows may
choose categories independently from the actual candidate. Ordinary `optimize_acqf_mixed`
remains invalid for this one-shot case because it fixes categorical features across the complete
augmented batch. Larger `q` and assignment spaces beyond the configured enumeration limit are
explicitly unsupported. See [Mixed one-shot optimization](mixed-one-shot-optimization.md).

These restrictions are deliberate: unsupported posterior shapes should fail clearly rather than
silently reducing the wrong dimension.


## Compatibility contract

The acquisition registry is intentionally a compatibility surface, not a catalog of every
BoTorch acquisition. Registered BoTorch entries cover representative standard MC, lookahead,
multi-objective, constrained, and multi-fidelity workflows. Native acquisitions documented above
but absent from the registry remain BoTorch-owned and are not implicitly recommended by the
capability selector.

Static compatibility enforces posterior sampling, fantasy, multi-fidelity, ensemble,
structured-output, and output-arity requirements. In particular, an acquisition that does not
support multi-output posteriors must reject a model whose public contract is multi-output-capable;
checking only `TaskType.MULTITASK` is insufficient because `ModelListGP` is multi-output without
being a multitask model.

The current metadata remains deliberately conservative for qKG and qMFKG: they are registered as
single-output acquisition workflows. Scalarized or custom-objective extensions should be added
only with executable integration tests and corresponding metadata changes.

Constraint support in the registry means the acquisition has a supported BoTorch composition
path; it does not mean robotorchan constructs constraint callables or objectives automatically.
Likewise, `supports_ensemble` means a compatible sampler/composition can be supplied, not that
every acquisition uses an empirical-ensemble sampler by default. Empirical non-GP ensemble
posteriors require `IndexSampler`; Gaussian posteriors, including MAP-SAAS model ensembles that
still expose a Gaussian posterior, use `SobolQMCNormalSampler`. The non-GP acquisition validator enforces this distinction for explicitly supplied samplers, while
preserving BoTorch's lazy `IndexSampler` initialization when the acquisition sampler is `None`.

## Final integration rule

New acquisition functionality should first check whether BoTorch already exposes the required
algorithm. Native functionality should be integration-tested and documented instead of wrapped.
A robotorchan implementation is justified when it supplies a genuinely missing algorithm or a
robotorchan-specific contract.


## Future extensions

The current integration surface covers the scoped single-objective BO,
regression active learning, level-set learning, information-theoretic BO, lookahead,
multi-objective BO, and multi-fidelity/cost-aware integration.

Useful future extensions are intentionally left as separate work rather than implied support:

- integration coverage for Hypervolume Knowledge Gradient in multi-objective BO;
- constrained/custom-objective multi-objective acquisition examples;
- explicit ensemble reduction semantics for robotorchan-specific regression acquisitions;
- optional multi-fidelity MES coverage where it provides value beyond MF-KG.

These are extension items, not compatibility gaps in the documented current contracts.

### Ensemble / non-GP contract

Posterior sampling type, not the word "ensemble" in a model name, controls sampler selection.
`make_model_sampler` maps Gaussian posteriors to `SobolQMCNormalSampler` and empirical ensemble
posteriors to `IndexSampler`. Runtime coverage includes both MAP-SAAS Gaussian ensembles and
tree-based empirical ensembles.

Robotorchan-specific active-learning acquisitions continue to reject empirical ensemble
posteriors explicitly. Defining variance, boundary, or predictive-information scores across
ensemble members requires an intentional reduction semantics; silently treating empirical
spread as Gaussian posterior variance would change the acquisition meaning. Ensemble AL remains
a future extension until that semantics is specified and tested.


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

### Correspondence status

The theory chapters, practical optimization guides, status table, package exports, and public-API
test are consistent with the current ownership policy. Remaining items listed under
**Audit follow-ups** are intentionally unsupported or not yet integration-tested rather than
silently implied by the theory documentation.

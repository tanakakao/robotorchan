# Preference acquisition selection (Phase 5)

This document records the Phase 5 selection decision for the current
single-utility, pairwise-comparison workflow. It does not extend the
public API or claim that generic active-learning acquisition functions
are compatible with preference posteriors.

## Scope and decision

| Approach | Existing integration | Phase 5 decision | Reason |
| --- | --- | --- | --- |
| Analytic EUBO | BoTorch `AnalyticExpectedUtilityOfBestOption`; Phase 3 and 4 tests | **Keep as default** | Directly proposes a comparison pair (`q=2`) using `PairwiseGP`. |
| Posterior-mean ranking | `PairwiseGP.posterior()` | **Keep as evaluation/baseline** | Ranks alternatives but does not actively select informative comparisons. |
| Monte Carlo best-option utility | No robotorchan preference-specific integration | **Defer** | No requirement beyond the current analytic, two-option workflow has been demonstrated. Verify upstream version and API before adding. |
| Preference-specific information gain / disagreement | No robotorchan preference-specific integration | **Defer** | Would require a separate preference observation model and validation of the acquisition objective. |
| Classification BALD / predictive entropy | Existing classification acquisition family | **Do not reuse directly** | Binary class-label uncertainty is not interchangeable with pairwise preference utility. |
| Multiobjective / outcome-model BOPE | Outside current preference scope | **Out of scope** | Requires a separate design and data contract. |

## Supported workflow

The current minimal path is:

1. Fit `robotorchan.models.PairwiseGP(datapoints, comparisons)`.
2. Build BoTorch `AnalyticExpectedUtilityOfBestOption(pref_model=model)`.
3. Optimize the acquisition with BoTorch `optimize_acqf(q=2)`.
4. Collect the winner/loser observation, append candidates and comparison indices.
5. Rebuild and refit on the accumulated observations.

The Phase 3 integration test and Phase 4 two-round loop test cover this
composition. Do not add a parallel `robotorchan` acquisition wrapper
without a concrete difference in behavior or user-facing need.

## Conditions for future extension

Reconsider an alternative only when it provides a concrete capability
not covered by analytic EUBO, such as a justified preference-specific
information-gain objective or a validated larger comparison set.
Confirm the API and compatibility against the pinned BoTorch version
before implementing it. Keep tests limited to the newly supported path.

## Phase 5 completion criterion

**No additional acquisition implementation is required for the current
q=2 preference workflow.** Proceed to Phase 6 benchmarking of the
existing EUBO loop rather than expanding the acquisition surface.

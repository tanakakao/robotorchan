# Combinatorial BO benchmark — Phase 9

Run from the repository root:

```bash
python examples/combinatorial_bo_benchmark.py --seeds 3 --steps 5 --dimensions 4
```

The script writes one CSV row per seed, method, and post-initialization
evaluation. Each method receives the same randomly selected initial points
per seed, a common objective, and the same objective-evaluation budget.
Report **best-so-far objective versus evaluations** and **wall time**
separately; wall time includes surrogate fitting and candidate search.

## Included methods

| Method | Surrogate | Candidate selection |
| --- | --- | --- |
| Random | None | Uniform among unseen points |
| Sobol | None | Scrambled Sobol threshold, nearest unseen point |
| Numeric GP baseline | BoTorch SingleTaskGP on binary numeric inputs | Enumerated UCB |
| Mixed GP baseline | MixedSingleTaskGP with all dimensions categorical | Enumerated UCB |
| Mixed GA baseline | MixedSingleTaskGP with all dimensions categorical | Mixed GA acquisition search; unseen fallback |
| Hamming local search | COMBO graph GP | Existing HammingLocalSearchStrategy |
| Discrete TR | COMBO graph GP | Existing DiscreteTrustRegionStrategy |
| BOCS | BOCSSurrogate (Gaussian prior) | Existing enumerated UCB |
| COMBO | COMBOSurrogate | Existing enumerated UCB |

The **numeric GP baseline** treats bits as numerical coordinates. The **Mixed GP**
baseline uses the actual `MixedSingleTaskGP` with all bits categorical.
The **Mixed GA** baseline uses the same mixed GP surrogate, optimizing its UCB
through the existing GA dispatch. When GA returns an observed point, the script
falls back to the best unseen point under UCB; this fallback is exhaustive and
must be reported when interpreting optimizer cost.
The local-search and trust-region variants use the COMBO surrogate, so the
comparison isolates acquisition search behavior, not different surrogates.

## Interpretation and remaining scope

This is a small binary, noiseless, single-objective q=1 **smoke benchmark**,
not a statistically powered algorithm ranking. The default objective is a
deceptive binary landscape; it is not representative of materials problems.
Use more objectives, dimensions, seeds, and confidence intervals before
drawing performance conclusions. Default dimensions are intentionally small
because BOCS/COMBO enumerate the entire binary domain.

The original Phase 9 method matrix now includes **Mixed GA** and the genuine
**MixedSingleTaskGP** baseline. The script is still a small smoke benchmark,
not a statistically powered performance study. COMBO here means the implemented
graph-GP/UCB variant, not the complete original COMBO algorithm.

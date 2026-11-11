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
| Mixed GP baseline | BoTorch SingleTaskGP | Enumerated UCB |
| Hamming local search | COMBO graph GP | Existing HammingLocalSearchStrategy |
| Discrete TR | COMBO graph GP | Existing DiscreteTrustRegionStrategy |
| BOCS | BOCSSurrogate (Gaussian prior) | Existing enumerated UCB |
| COMBO | COMBOSurrogate | Existing enumerated UCB |

The **Mixed GP baseline** is a GP on binary-valued numeric coordinates, not
`MixedSingleTaskGP`; this distinction must be preserved in result reports.
The local-search and trust-region variants use the COMBO surrogate, so the
comparison isolates acquisition search behavior, not different surrogates.

## Interpretation and remaining scope

This is a small binary, noiseless, single-objective q=1 **smoke benchmark**,
not a statistically powered algorithm ranking. The default objective is a
deceptive binary landscape; it is not representative of materials problems.
Use more objectives, dimensions, seeds, and confidence intervals before
drawing performance conclusions. Default dimensions are intentionally small
because BOCS/COMBO enumerate the entire binary domain.

The originally planned **Mixed GA** and **MixedSingleTaskGP** baselines are
not yet wired into this script. They should be added before declaring the
full Phase 9 comparison matrix complete. COMBO here means the implemented
graph-GP/UCB variant, not the complete original COMBO algorithm.

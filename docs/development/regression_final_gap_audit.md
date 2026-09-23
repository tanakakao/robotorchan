# Regression final gap audit — Phase 17

This audit uses the current Phase 16 branch as source of truth. Classification and ordinal models
remain outside this stream.

## Current result

The regression surface has no newly identified broad correctness defect. The main remaining work is
concentrated in Kronecker extensions and a small number of deliberately research-gated
cross-capability models. Validation-only gaps should not be confused with missing statistical
models.

| Candidate / area | Decision | Priority | Reason |
| --- | --- | --- | --- |
| Heteroskedastic Kronecker | Prototype | high | Valuable block-design noise model, but the current prototype predicts task-specific noise without coupling it into the response observation likelihood. |
| Mixed heteroskedastic Kronecker | Hold behind continuous prototype | high | Mixed covariance is already available, but public semantics depend on solving the continuous shaped-noise likelihood first. |
| Spectral-mixture Kronecker | Implement | high | The existing spectral-mixture MultiTask family already isolates the data kernel; a block-design Kronecker variant can preserve output-task covariance while replacing only the input covariance. |
| Infinite-width BNN Kronecker | Implement | medium-high | The NNGP kernel is an exact GP covariance and can serve as the Kronecker data kernel without changing block-design task semantics. |
| SAAS / fully Bayesian Kronecker | Prototype | medium-high | High-dimensional shrinkage is valuable, but current BoTorch fully Bayesian multitask support is long-format; a block-design posterior / MCMC ownership boundary must be designed explicitly. |
| DeepGP Kronecker | Hold | medium | Deep latent composition plus block-design task covariance requires a substantially different variational design; simple covariance substitution is not sufficient. |
| RRP Kronecker | Prototype | medium | Requires a shaped multitask sparse-outlier observation model rather than reuse of the single-task noise-covar mixin. |
| Student-t Kronecker | Prototype | medium | Statistically meaningful, but needs a block-design variational likelihood/posterior contract. |
| Contaminated Kronecker | Prototype | medium | Requires an explicit task-shared versus task-specific contamination contract. |
| Replicate-noise Kronecker | Prototype | medium | Practical for aligned repeated measurements, but grouping and task-specific variance-of-mean semantics must be defined. |
| Ensemble MAP-SAAS MultiFidelity | Hold | medium | Valuable but still lacks a stable ensemble covariance / GaussianMixturePosterior ownership seam. |
| RRP MultiFidelity | Hold | medium | Fidelity covariance and sparse observation-outlier semantics still need an intentional composition. |
| Additional external non-GP regression model | Hold | low | NGBoost plus empirical tree/boosting surrogates cover the current practical external-surrogate need; another backend is not justified by a current capability gap. |

## Kronecker family conclusion

The Kronecker family is not considered finished. The next implementation wave should first add
expressive variants whose statistical composition is clean:

1. `SpectralMixtureKroneckerMultiTaskGP`;
2. `InfiniteWidthBNNKroneckerMultiTaskGP`.

Both candidates keep the standard block-design contract:

- `train_X[..., n, d]` contains only data features;
- `train_Y[..., n, m]` contains aligned task outputs;
- the expressive kernel replaces only `data_covar_module`;
- output-task covariance remains owned by `KroneckerMultiTaskGP`;
- raw training tensors and `make_mll` remain unchanged.

Each implementation must prove posterior shape, finite samples, scalarized MC acquisition, candidate
optimization where differentiable, and public registry / documentation alignment. Mixed variants
should follow only after the continuous variant passes these gates.

## Heteroskedastic Kronecker recovery gate

The current heteroskedastic Kronecker prototype is not a completed public model. Re-publication
requires a response observation model whose shaped noise has both input and task axes and is
actually consumed by the likelihood / posterior. Merely fitting an auxiliary noise GP is
insufficient.

A future implementation must demonstrate, with an executable test, that changing learned
task-specific noise changes the response observation covariance while preserving the latent
Kronecker data × task covariance.

## Exit assessment

Phases 1–16 substantially closed the original validation, robust MultiTask, Mixed robust,
MultiFidelity × high-dimensional, MultiFidelity × robust, external probabilistic non-GP,
acquisition, optimizer, registry, and documentation goals.

Phase 17 therefore does **not** close the development stream. It converts the remaining work into a
smaller extension backlog. Phase 18 should begin the expressive Kronecker implementation wave,
starting with spectral-mixture Kronecker because it has the cleanest covariance composition and
direct reuse of the existing spectral-mixture kernel contract.

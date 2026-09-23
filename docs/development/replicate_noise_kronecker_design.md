# Replicate-noise x Kronecker design audit

## Decision

Phase 11 classifies ReplicateNoise x Kronecker as **Prototype / blocked by observation-noise
integration**, rather than adding a public model.

Replicate aggregation itself has clean block-design semantics. The blocker is the same response
likelihood boundary identified by the heteroskedastic Kronecker work: the current public
`KroneckerMultiTaskGP` path does not accept an explicit fixed `n × m` observation-variance
matrix while preserving its block-design posterior semantics.

A public class must not silently collapse the task-specific empirical variances to one scalar,
one variance per design row, or a learned multitask likelihood parameter.

## Replicate contract

Raw replicate data use

- `train_X[r, d]`;
- `train_Y[r, m]`;

where duplicate rows of `train_X` identify aligned replicate groups. Every group must contain at
least two observations.

For group `g`, task `t`, and replicate count `R_g`:

`mean[g, t] = (1 / R_g) sum_r y[g, r, t]`

`replicate_variance[g, t] = Var_r(y[g, r, t])`

`mean_variance[g, t] = replicate_variance[g, t] / R_g`

The GP training contract after aggregation is therefore

- `unique_X[G, d]`;
- `group_mean[G, m]`;
- `mean_variance[G, m]`.

The final tensor is observation variance of the **group mean**, not raw replicate variance.

## Alignment semantics

Replicates are grouped by complete equality of each raw design row. A replicate group is shared
across all output tasks because block-design observations are aligned by construction.

The implementation must reject:

- `train_Y` without an explicit output-task axis;
- row-count mismatch between X and Y;
- non-finite X or Y;
- groups with fewer than two replicate rows;
- any later attempt to broadcast `G × 1` noise over `G × m` tasks.

Missing task observations inside a group are outside this contract. Supporting them would require
a partially observed multitask model rather than the aligned Kronecker design.

## Aggregation helper

A future runtime implementation should add a dedicated
`_aggregate_kronecker_replicates` helper instead of reusing the current single-task
`_aggregate_replicates` unchanged.

The helper should return

`unique_X, group_mean, mean_variance, counts, replicate_variance`

with shapes

- `unique_X: G × d`;
- `group_mean: G × m`;
- `mean_variance: G × m`;
- `counts: G`;
- `replicate_variance: G × m`.

The current single-task helper deliberately enforces `train_Y[..., 1]`; weakening that validation
globally would blur the public single-task contract.

## Statistical model

The latent response covariance remains

`K_f = K_data(X, X') ⊗ K_task`.

The empirical observation-noise operator should be

`D_rep = diag(vec(mean_variance))`.

The training covariance is

`K_y = K_f + D_rep`.

This is a fixed-noise exact Gaussian model. Unlike the heteroskedastic prototype, no second noise
GP is required because replicate observations directly estimate the variance of each group mean.

## Why this is not a scalar likelihood

For two tasks measured under the same design condition, replicate dispersion may differ. Therefore

`mean_variance[g, 0] != mean_variance[g, 1]`

is valid and expected.

Replacing this matrix with a scalar learned likelihood noise, a task-only noise vector, or the
mean across tasks changes the statistical model. Such a shortcut is not accepted as
ReplicateNoise x Kronecker.

## Required runtime acceptance test

Before a public `ReplicateNoiseKroneckerMultiTaskGP` is added, an executable test must prove:

1. replicate aggregation produces the expected `G × m` means and variance-of-mean values;
2. changing the replicate dispersion of task `t` changes the corresponding diagonal of the
   response observation covariance;
3. the latent covariance is unchanged by that observation-noise change;
4. another task's requested empirical variance is not silently changed;
5. posterior samples are finite and preserve the output task axis.

The covariance test is the key gate. Storing `mean_variance` as metadata is insufficient.

## Acquisition and optimizer gate

Once observation covariance is integrated correctly, public promotion additionally requires:

- scalarized MC `qLogExpectedImprovement`;
- `qUpperConfidenceBound`;
- standard posterior sampling;
- continuous `optimize_acqf`;
- `make_mll` / exact-Gaussian training-objective compatibility.

These are runtime requirements, not capability assumptions.

## Mixed extension

Mixed ReplicateNoise x Kronecker is meaningful only after the continuous model passes the
observation-noise gate. Replicate grouping in a future Mixed model must use the complete raw row,
including categorical values. Categorical dimensions remain data features and never encode task
identity.

No Mixed public class should be added first as a workaround.

## Public API decision

Phase 11 does not add:

- `ReplicateNoiseKroneckerMultiTaskGP`;
- a registry entry;
- capability metadata;
- model coverage;
- a compatibility alias.

The model remains a documented prototype target until robotorchan has a clean block-design
fixed-noise observation path. If the heteroskedastic Kronecker work later supplies that shared
infrastructure, ReplicateNoise x Kronecker should reuse it; it should not create a second private
posterior implementation.

## Revisit trigger

Promote this item from Prototype to Implement when the Kronecker response model can consume an
explicit task-specific `G × m` fixed-noise matrix through a maintainable public likelihood /
posterior contract.

This is an integration blocker, not a statistical objection to replicate-noise Kronecker models.

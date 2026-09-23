# MAP-SAAS × multi-fidelity covariance prototype

## Result

The Phase 23 source audit confirms that the **non-linear-truncated** BoTorch
`SingleTaskMultiFidelityGP` path is the appropriate composition seam for a future exact
MAP-SAAS multi-fidelity model.

BoTorch exposes `covar_module` as the covariance over **non-fidelity features**. Its native
multi-fidelity model then combines that module with `DownsamplingKernel` for data fidelity and
`ExponentialDecayKernel` for iteration fidelity. This is precisely the separation required by
the Phase 22 statistical contract.

## Prototype composition

The safe prototype is:

```text
raw X
  |
  +-- design dims ------> ARD design kernel + add_saas_prior
  |
  +-- data fidelity ----> DownsamplingKernel
  |
  +-- iteration fidelity -> ExponentialDecayKernel
  |
  +---------------------> SingleTaskMultiFidelityGP composition
```

The important point is that `add_saas_prior` is attached to the design `covar_module` before
it is passed to `SingleTaskMultiFidelityGP`. The native fidelity kernels never receive the SAAS
prior.

## Why no public model is added in Phase 23

There are two distinct BoTorch fidelity constructions:

1. `linear_truncated=False`: the model accepts a separate non-fidelity `covar_module` and
   combines it with native data/iteration fidelity kernels. This is compatible with design-only
   SAAS shrinkage.
2. `linear_truncated=True`: `LinearTruncatedFidelityKernel` internally owns several Matern
   kernels over non-fidelity dimensions. A single externally supplied SAAS design kernel does not
   represent that construction.

Therefore a first public MAP-SAAS multi-fidelity implementation should explicitly require
`linear_truncated=False`. Pretending to support both modes would overstate the statistical
contract.

## Required implementation contract

The first implementation should:

- normalize fidelity dimensions before constructing the design kernel;
- set the design kernel `active_dims` to every non-fidelity raw dimension;
- set `ard_num_dims` to the number of design dimensions only;
- apply BoTorch `add_saas_prior` to that design kernel only;
- pass that kernel through `SingleTaskMultiFidelityGP(covar_module=...)`;
- force `linear_truncated=False`;
- preserve raw training tensors;
- preserve separate iteration and data fidelity roles;
- expose enough metadata to verify design and fidelity dimensions;
- retain `make_mll()`, posterior sampling, and raw-space acquisition behavior.

## Executable assertions for the implementation phase

Tests must inspect the actual kernel tree rather than infer correctness from a finite posterior.
They should prove:

1. the design kernel has SAAS `tau` / lengthscale prior registration;
2. its ARD dimension count equals the number of non-fidelity features;
3. no fidelity kernel owns SAAS parameters or priors;
4. changing a fidelity coordinate is evaluated through native fidelity covariance;
5. iteration-only, data-only, and combined fidelity configurations construct successfully;
6. duplicate/overlapping fidelity roles are rejected;
7. negative fidelity indices normalize correctly;
8. posterior and `rsample` are finite;
9. exact MLL construction succeeds;
10. at least one true multi-fidelity acquisition evaluates.

## Additive and ensemble boundary

This prototype does not establish that BoTorch `AdditiveMapSaasSingleTaskGP` can be inserted as a
submodel. Its additive covariance has different semantics from a single SAAS-regularized design
kernel.

Likewise, `EnsembleMapSaasSingleTaskGP` is a batched ensemble with
`GaussianMixturePosterior` semantics. Passing an ensemble-shaped covariance into a
multi-fidelity model requires an independent batch/posterior contract audit.

The first public model should therefore be named for its actual semantics, not as an alias of
either existing MAP-SAAS wrapper.

## Fully Bayesian boundary

Nothing in this prototype changes the status of fully Bayesian SAAS × multi-fidelity. That path
still requires a dedicated fidelity-aware Pyro model.

## Phase 23 decision

**Proceed to implementation**, but only for the exact, non-linear-truncated composition:

```text
design-only ARD kernel + SAAS prior
                ×
BoTorch native fidelity covariance
```

Phase 24 should implement that contract and keep additive, ensemble, and fully Bayesian variants
out of the public API until their distinct semantics are validated.

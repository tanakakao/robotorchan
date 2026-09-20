# DeepGP variant policy

`SingleTaskDeepGP` is the supported Deep Gaussian Process model in the current
robotorchan release.

## Why there is no automatic Mixed DeepGP variant

A categorical coordinate must not be passed through a continuous DeepGP layer as
if its integer code had metric meaning. A useful mixed-input DGP therefore needs
an explicit architecture for categorical latent mappings or a principled mixed
kernel at every relevant stochastic layer. Reusing the exact-GP mixed kernel only
at the final layer would not preserve that semantics.

For this reason robotorchan does not add a nominal `MixedSingleTaskDeepGP`
wrapper merely for API symmetry.

## Multi-output and multitask

A multi-output DGP is mathematically meaningful, but it requires a deliberate
choice between independent output layers, shared latent processes, and correlated
task/output structure. These alternatives imply different posterior shapes and
BoTorch objective semantics.

The first DeepGP implementation therefore remains single-output until those
structures are implemented and tested as real models rather than aliases.

## Current integration contract

The supported model provides a Monte Carlo posterior for BoTorch acquisition
functions. The posterior preserves stochastic DeepGP trajectories instead of
collapsing them to an exact Gaussian posterior. Batched candidate evaluation keeps
t-batch dimensions when trajectories are selected from normal base samples.

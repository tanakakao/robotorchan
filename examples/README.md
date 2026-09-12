# robotorchan examples

This directory contains executable examples for the public models exposed by `robotorchan.models`.

The primary examples are Jupyter notebooks under `examples/notebooks/`.

## Installation

Install robotorchan with the example dependencies:

```bash
pip install -e ".[examples]"
```

For fully Bayesian SAAS examples, also install the optional fully Bayesian dependencies:

```bash
pip install -e ".[examples,fully-bayesian]"
```

## Notebook design rules

Each notebook should be self-contained and executable from top to bottom.

Use the following section structure where applicable:

1. When to use this model
2. Imports and reproducibility setup
3. Synthetic example data
4. Model construction
5. robotorchan common API
6. Model fitting
7. Posterior prediction
8. Visualization
9. Bayesian optimization example, when appropriate
10. When to use / when not to use

### Common API demonstration

For supervised wrappers, examples should demonstrate the robotorchan-specific common API where supported:

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.raw_data_names
model.supports_mll
model.make_mll()
```

Specialized models should demonstrate their corresponding raw-data interface rather than forcing the supervised convention. For example, `PairwiseGP` uses pairwise-comparison data and fully Bayesian SAAS models intentionally do not expose exact-MML training through `make_mll()`.

## Data policy

Examples should use small synthetic datasets unless a real dataset is essential to explain the model.

Prefer synthetic data because it keeps notebooks:

- reproducible,
- fast to execute,
- independent of network access,
- easy to understand,
- suitable for future CI execution.

Avoid hidden helper modules for core example logic. A notebook should remain understandable when copied and executed on its own. Small local helper functions inside a notebook are preferred over shared example utilities when the duplication is minor.

## Plotting

Use Matplotlib for lightweight visualization. Avoid adding optional plotting dependencies unless they materially improve a specific example.

Typical regression notebooks should visualize, where practical:

- training observations,
- posterior mean,
- posterior uncertainty,
- selected candidate points for BO examples.

## Notebook index

| Notebook | Main models | Status |
|---|---|---|
| [`01_single_task_gp.ipynb`](notebooks/01_single_task_gp.ipynb) | `SingleTaskGP` | Available |
| [`02_mixed_single_task_gp.ipynb`](notebooks/02_mixed_single_task_gp.ipynb) | `MixedSingleTaskGP` | Available |
| [`03_multi_fidelity_gp.ipynb`](notebooks/03_multi_fidelity_gp.ipynb) | `SingleTaskMultiFidelityGP` | Available |
| [`04_multitask_gp.ipynb`](notebooks/04_multitask_gp.ipynb) | `MultiTaskGP`, `KroneckerMultiTaskGP` | Available |
| [`05_model_list_gp.ipynb`](notebooks/05_model_list_gp.ipynb) | `ModelListGP` | Available |
| [`06_variational_gp.ipynb`](notebooks/06_variational_gp.ipynb) | `SingleTaskVariationalGP` | Available |
| [`07_pairwise_gp.ipynb`](notebooks/07_pairwise_gp.ipynb) | `PairwiseGP` | Available |
| [`08_saas_gp.ipynb`](notebooks/08_saas_gp.ipynb) | `SaasFullyBayesianSingleTaskGP`, `SaasFullyBayesianMultiTaskGP` | Available |
| [`09_map_saas_and_additive_gp.ipynb`](notebooks/09_map_saas_and_additive_gp.ipynb) | `AdditiveMapSaasSingleTaskGP`, `EnsembleMapSaasSingleTaskGP`, `OrthogonalAdditiveGP` | Available |
| [`10_robust_gp.ipynb`](notebooks/10_robust_gp.ipynb) | `RobustRelevancePursuitSingleTaskGP` | Available |
| [`11_structured_output_gp.ipynb`](notebooks/11_structured_output_gp.ipynb) | `HigherOrderGP`, `LatentKroneckerGP` | Available |
| [`12_hierarchical_gp.ipynb`](notebooks/12_hierarchical_gp.ipynb) | `HierarchicalConditionalKernelGP`, `HierarchicalConditionalKernelMultiTaskGP` | Available |
| `13_heterogeneous_multitask_gp.ipynb` | `HeterogeneousMTGP` | Planned |
| `14_contextual_gp.ipynb` | `SACGP`, `LCEAGP`, `LCEMGP` | Planned |

## Reproducibility

Notebook examples should set a fixed PyTorch seed near the beginning:

```python
import torch

torch.manual_seed(0)
```

Prefer `torch.double` for GP examples unless there is a model-specific reason to use another dtype.

## Runtime expectations

Most notebooks should remain lightweight enough for routine execution.

Fully Bayesian NUTS examples are an exception and may require reduced sampling settings for documentation or smoke-test use. They should be kept separate from the normal lightweight notebook execution path when CI support is added.

## Relationship to documentation and tests

- `docs/models.md`: explains which model to choose and why.
- `examples/notebooks/`: shows how to run each model.
- `tests/`: verifies implementation behavior and compatibility.

The notebooks are documentation examples, not replacements for unit tests.

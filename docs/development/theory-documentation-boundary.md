# Theory documentation boundary and migration plan

## Purpose

Keep `docs/theory/` independent of robotorchan-specific implementation and public APIs.
This document defines ownership and a migration map. Phase 2 establishes the
boundary; subsequent phases migrate content and validate mathematical claims.

## Ownership

| Content | Canonical home |
| --- | --- |
| Mathematical definitions, assumptions, derivations, and theoretical limitations | `docs/theory/` |
| General mathematical algorithms, numerical properties, and asymptotic complexity | `docs/theory/` |
| Model names, constructors, arguments, fit/predict usage, and supported combinations | `docs/models/` |
| Acquisition/optimizer configuration, dispatch, supported constraints, and examples | `docs/optimization/` |
| Internal contracts, implementation decisions, capabilities, and backend compatibility | `docs/development/` |
| Executable demonstrations and workflows | `examples/notebooks/` |

A paper's algorithm and its mathematical pseudocode are theoretical content.
A named library backend, implementation status, API parameter, or runtime behavior
is implementation documentation. Cross-links are permitted; duplicated ownership
is not.

## Chapter contract

A theory chapter should contain, where relevant:

1. Problem formulation and notation.
2. Assumptions and domain of validity.
3. Mathematical definitions and derivations.
4. Statistical or algorithmic properties, including computational complexity.
5. Relations to alternative methods and limitations.
6. Primary references.

Avoid package imports, constructor signatures, parameter names specific to
robotorchan, compatibility matrices, and claims about currently supported
features. Distinguish a mathematical model from its numerical approximation.

## Migration map

| Source | Retain in theory | Move to implementation documentation |
| --- | --- | --- |
| `docs/theory/README.md` | Conceptual map and chapter links | Library correspondence and capability matrices |
| `docs/theory/05_mixed_variables.md` | Categorical covariance and mixed-domain geometry | `MixedSingleTaskGP` usage and encoding contracts to `docs/models/` |
| `docs/theory/07_multitask_multioutput.md` | Task covariance, ICM, Kronecker algebra | Named model configuration and `task_feature` API to `docs/models/` |
| `docs/theory/12_hierarchical_contextual_gp.md` | Conditional kernels and hierarchical spaces | Constructors, arguments, `eval_hierarchical_features` to `docs/models/` |
| `docs/theory/13_model_selection.md` | Statistical model selection criteria and assumptions | Library model selection workflow to `docs/models/` |
| `docs/theory/17_nonstationary_gp.md` | Input-dependent lengthscales and nonstationarity | robotorchan-specific parameterization to `docs/models/` |
| `docs/theory/18_dimensionality_reduction_gp.md` | Projection and reduction mathematics | Reducer lifecycle and transform ordering to `docs/models/` or `docs/development/` |
| `docs/theory/24_classification.md` | Latent and predictive distributions, inference, uncertainty | Model family support and prediction contracts to `docs/models/` |
| `docs/theory/acquisition/README.md` | Mathematical taxonomy and navigation | Public acquisition names and compatibility to `docs/optimization/` |
| `docs/theory/acquisition/08_constraints.md` | Feasibility, outcome and candidate constraints | API-specific sign conventions and dispatch to `docs/optimization/` |
| `docs/theory/acquisition/12_selection_guide.md` | Decision-theoretic method comparisons | Current support, named classes, runtime advice to `docs/optimization/` |
| `docs/theory/optimization/README.md` | Mathematical optimization taxonomy | Backend, orchestration, capability metadata to `docs/optimization/` |
| `docs/theory/optimization/01_foundations.md` | Acquisition maximization and SAA | Capability and implementation mapping to `docs/optimization/` |
| `docs/theory/optimization/02_gradient_based.md` | Gradients, reparameterization, multistart | BoTorch/Torch backend specifics to `docs/optimization/` |
| `docs/theory/optimization/09_high_dimensional.md` | REMBO, HeSBO, ALEBO, BAxUS geometry | `LatentSpaceStrategy` implementation to `docs/optimization/` |
| `docs/theory/optimization/12_selection_guide.md` | Mathematical algorithm trade-offs | Dispatcher, GPU, named backends, MixedVariableSpace, capability tables to `docs/optimization/` |

The remaining chapters are subject to the same rules. This table identifies
priority examples, not an assertion that other chapters have no implementation
content.

## Overlap boundaries

- `04_acquisition_function.md` is the introduction; `acquisition/` owns derivations.
- `06_multi_fidelity.md` owns surrogate theory; `acquisition/11_multifidelity_cost_aware.md` owns information/cost decisions.
- `08_high_dimensional_gp.md` introduces high-dimensional statistical issues;
  `21_advanced_high_dimensional_models.md` owns advanced priors and variants.
- `19_neural_reduction_gp.md` owns dimension-reducing representations;
  `22_expressive_gp.md` owns expressive kernels and deep GP distributions.
- `20_high_dimensional_search.md` provides the overview;
  `optimization/09_high_dimensional.md` owns search geometry and algorithms.
- `24_classification.md` owns predictive probability and likelihood;
  `acquisition/09_active_learning.md` and classification AL chapters own selection criteria.

Do not delete or merge chapters solely because concepts recur. Preserve
mathematical detail, citations, and existing external links.

## Acceptance criteria for later migration

- Every relocated implementation statement has a canonical destination.
- No mathematical derivation is lost in a move.
- Theory chapters have no package-specific API instructions or capability claims.
- Internal relative links and documentation checks pass.
- Equations, notation, assumptions, and references are reviewed separately
  from documentation relocation.

## Review safeguards

Before declaring the model-theory migration complete, compare source and
result for large chapter rewrites. Confirm unique derivations, assumptions,
counterexamples, and references are preserved or moved to their canonical
owners. Passing link checks alone does not prove content preservation.

Keep terminology and language consistent across chapters, and validate the
integrated default branch after independent documentation changes are merged.

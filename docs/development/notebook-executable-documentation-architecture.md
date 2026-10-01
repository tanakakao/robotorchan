# Notebook executable documentation architecture

## Purpose

This document defines the information architecture for executable documentation in
`examples/notebooks/`. It is intentionally organized by learning responsibility rather than by
public class count.

The source-of-truth relationship is:

```text
Theory
  -> executable concept example
  -> cross-layer workflow
  -> tests / benchmarks
```

A notebook demonstrates usage and semantics. It does not replace unit tests, capability metadata,
or benchmarks.

## Design principles

1. Keep model-family notebooks concept-oriented. Do not create one notebook per public class.
2. Add explicit cross-layer workflow notebooks for complete BO and active-learning pipelines.
3. Do not force every model into the same pipeline when its posterior or optimization semantics
   differ.
4. Link notebooks to the relevant theory chapter and runtime-facing guide.
5. Keep ordinary CI notebooks CPU-friendly and deterministic.
6. Keep expensive MCMC, structured-output, and benchmark examples in an explicit slow tier.
7. Prefer public `robotorchan` and BoTorch APIs over notebook-only helper abstractions.

## Target taxonomy

The target logical taxonomy is:

```text
examples/notebooks/
  getting_started/
  models/
  acquisition/
  optimization/
  workflows/
```

The physical move is deferred until links, CI discovery, and redirects/document references can be
changed atomically. During the migration, the existing numbered notebooks remain valid.

### getting_started

Small notebooks that teach the minimum end-to-end contract:

```text
data -> model -> fit -> posterior -> acquisition -> optimization -> candidate
```

This is the shortest executable route for a new user.

### models

Model-family notebooks explain model construction, training contract, posterior semantics, and
family-specific caveats. They may include candidate generation when it is pedagogically useful,
but full BO-loop coverage is not mandatory.

### acquisition

Executable examples for acquisition semantics independent of any single model family:

- analytic versus Monte Carlo acquisition
- posterior sampling
- objective and PosteriorTransform
- single- and multi-objective acquisition
- constraints
- active learning
- cost-aware / multi-fidelity acquisition
- batch, pending points, and fantasization

### optimization

Executable examples for acquisition optimization and candidate-search geometry:

- continuous optimization
- mixed / discrete optimization
- linear and nonlinear constraints
- initialization
- derivative-free and evolutionary strategies
- high-dimensional search
- trust regions
- hybrid strategies

### workflows

Complete executable workflows that connect multiple layers. These are the primary integration
examples and should make layer boundaries visible instead of hiding them in helpers.

Initial workflow set:

1. sequential single-objective BO
2. batch / asynchronous BO
3. mixed-variable BO
4. constrained BO
5. multi-objective BO
6. multi-fidelity / cost-aware BO
7. MultiTask / Kronecker BO
8. high-dimensional BO
9. robust / uncertain-input BO where the model contract supports it
10. regression active learning

## Notebook tiers

### Tier A: full pipeline

A Tier A notebook should normally demonstrate:

```text
Data
 -> Model
 -> Fit
 -> Posterior
 -> Sampling when required
 -> Objective / PosteriorTransform when required
 -> Acquisition
 -> Acquisition optimization
 -> Candidate
 -> Evaluation
```

Tier A is required for the representative workflows, not for every model class.

### Tier B: model family

Tier B demonstrates:

```text
Data -> Model -> Fit -> Posterior -> family-specific semantics
```

It should link to a Tier A workflow showing the nearest compatible end-to-end use.

### Tier C: specialist

Tier C is for models whose semantics do not fit the standard regression BO pipeline without
important qualifications, including preference learning, expensive fully Bayesian examples,
structured outputs, and selected expressive or non-GP models.

A Tier C notebook must state what is intentionally omitted and link to the relevant theory and
compatibility documentation.

## Existing notebook disposition

| Existing notebook | Disposition | Target role |
| --- | --- | --- |
| 01 SingleTaskGP | Expand | Tier A getting-started reference |
| 02 MixedSingleTaskGP | Expand | Tier A mixed-variable reference |
| 03 MultiFidelityGP | Expand | Tier A/B multi-fidelity reference |
| 04 MultiTask / Kronecker | Expand | Tier A/B MultiTask reference |
| 05 ModelListGP | Expand | Tier B multi-output model example; connect to MO workflow |
| 06 VariationalGP | Keep + link | Tier B scalable-model example |
| 07 PairwiseGP | Specialist | Tier C preference-learning example |
| 08 SAAS | Specialist | Tier C expensive fully Bayesian example |
| 09 MAP-SAAS / Additive | Keep + link | Tier B high-dimensional model example |
| 10 Robust GP | Expand | Tier B robust-model example |
| 11 Structured output | Specialist | Tier C structured-output example |
| 12 Hierarchical GP | Keep + link | Tier B structured-input example |
| 13 Heterogeneous MultiTask | Keep + link | Tier B structured-task example |
| 14 Contextual GP | Keep + link | Tier B contextual-model example |
| 15 Robust observation | Merge conceptually | Tier B robust family; share workflow links with 10 |
| 16 Noise models | Merge conceptually | Tier B noise family; connect to robust workflow |
| 17 Uncertain input | Expand | Tier B uncertain-input example |
| 18 Nonstationary GP | Keep + link | Tier B nonstationary example |
| 19 Linear reduction | Keep + link | Tier B representation example |
| 20 Neural reduction | Specialist | Tier C representation-learning example |
| 21 Reduced MultiTask | Keep + link | Tier B high-dimensional MultiTask example |
| 22 Mixed reduction | Keep + link | Tier B mixed high-dimensional example |
| 23 High-dimensional benchmark | Replace role | benchmark explainer; workflow lives separately |
| 24 Expressive surrogate | Split conceptually | Tier C overview plus focused runnable examples as needed |
| 25 Non-GP surrogates | Split conceptually | Tier C model semantics plus compatible workflow examples |

"Merge conceptually" does not mean deleting notebooks immediately. It means the notebooks share one
documentation family and should avoid duplicating theory. "Split conceptually" means one overview
notebook should not pretend that models with different posterior semantics share an identical
training or acquisition contract.

## Theory-to-notebook mapping

| Theory area | Executable documentation target |
| --- | --- |
| 01 Bayesian Optimization | getting-started + sequential workflow |
| 02 Gaussian Process | SingleTaskGP model notebook |
| 03 Kernel | focused model sections; no kernel-per-notebook explosion |
| 04 + acquisition/* | acquisition notebooks + workflows |
| 05 Mixed Variables | mixed model + mixed workflow |
| 06 Multi-Fidelity | multi-fidelity model + cost-aware workflow |
| 07 MultiTask / Multi-output | MultiTask, ModelList + MultiTask/MO workflows |
| 08, 18-21 High dimensional | model examples + high-dimensional workflow |
| 09 Variational | VariationalGP model notebook |
| 10 Preference | Pairwise specialist notebook |
| 11 Structured Output | structured-output specialist notebook |
| 12 Hierarchical / Contextual | corresponding model notebooks |
| 14-17 Robustness | robust/noise/uncertain-input model family + compatible workflow |
| 22 Expressive GP | specialist model examples |
| 23 Non-GP | posterior-semantics examples + compatible workflow |
| optimization/* | optimization notebooks + workflows |

Each notebook should link to the smallest relevant theory chapter rather than only to the theory
index.

## Cross-layer coverage contract

Coverage is tracked by concept, not by notebook count. The executable documentation matrix uses
these columns:

- model construction
- fit / inference
- posterior
- sampling
- objective / PosteriorTransform
- acquisition
- initialization
- candidate optimization
- constraints
- pending / fantasy
- evaluation / loop update
- theory link
- runtime guide link
- CI tier

A blank cell means "not demonstrated here", not "unsupported by robotorchan". Runtime support must
continue to come from capability metadata, implementation validation, and tests.

## Migration order

1. Add a coverage manifest and validate notebook metadata.
2. Upgrade the SingleTaskGP notebook into the canonical minimum end-to-end example.
3. Add workflow notebooks for sequential, mixed, constrained, multi-objective, and batch/async BO.
4. Add acquisition-focused notebooks.
5. Add optimization-focused notebooks.
6. Connect MultiFidelity, MultiTask/Kronecker, high-dimensional, robust, and active-learning
   workflows.
7. Add theory/runtime-guide backlinks.
8. Move notebooks into physical subdirectories only after all links and CI discovery are ready.
9. Run notebook CI and link validation before removing legacy paths.

## Non-goals

- one notebook per public class
- duplicating benchmark implementations inside notebooks
- treating every posterior as Gaussian
- forcing every model through `optimize_acqf`
- hiding cross-layer contracts behind a large notebook helper framework
- using notebook execution as a substitute for unit or integration tests

## Phase 2 decision

The current 25 notebooks are retained as migration inputs. No notebook is deleted in this phase.
The target architecture is model-family documentation plus explicit acquisition, optimization, and
workflow documentation. Physical reorganization is delayed until the migration can preserve links
and CI atomically.

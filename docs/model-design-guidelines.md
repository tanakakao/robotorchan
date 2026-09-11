# Model design guidelines

This document defines the default design rules for model extensions and BoTorch wrappers in `robotorchan`.

The goal is to keep the library predictable for users who already know BoTorch while still allowing robotorchan-specific conveniences such as raw-data retention, `make_mll()`, mixed-input support, and research-oriented extensions.

These rules are defaults. A model-specific exception is acceptable only when the upstream BoTorch architecture makes the default rule unsafe or misleading, and the exception should be documented and tested.

## 1. BoTorch compatibility is the default

Public model APIs should follow BoTorch as closely as practical.

- Preserve upstream class naming, argument names, argument order, defaults, tensor conventions, return types, posterior semantics, and exception behavior where possible.
- Prefer thin subclasses / wrappers around existing BoTorch models rather than replacing their behavior.
- Keep standard BoTorch workflows valid, including `fit_gpytorch_mll`, `fit_fully_bayesian_model_nuts`, acquisition functions, posterior calls, conditioning, and fantasy-model workflows when supported upstream.
- Do not introduce a robotorchan-specific abstraction when the native BoTorch abstraction already solves the problem cleanly.
- When extending an upstream constructor, preserve the existing constructor surface and add only the minimum new arguments required by the extension.
- Existing non-mixed wrappers should not gain categorical-only arguments merely because a Mixed variant exists.

Compatibility is behavioral, not only syntactic. A wrapper that has the same constructor but silently changes model semantics is not considered compatible.

## 2. Monkey patching is prohibited

Do not modify BoTorch, GPyTorch, PyTorch, or third-party objects at runtime.

Prohibited patterns include:

- replacing or adding attributes on imported BoTorch / GPyTorch classes or functions;
- assigning replacement implementations into imported modules;
- modifying `sys.modules` to redirect imports;
- import-time patches whose behavior depends on import order;
- globally replacing methods such as `posterior`, `forward`, fitting utilities, or optimizers;
- hidden process-wide state changes used to make a wrapper work.

Implement extensions explicitly inside robotorchan using subclasses, wrappers, adapters, transforms, helper functions, or composition.

A user importing BoTorch directly must observe normal upstream BoTorch behavior regardless of whether robotorchan has been imported.

## 3. Prefer thin wrappers over duplicated upstream implementations

When an existing BoTorch model already provides the required mathematical model, robotorchan should normally wrap or subclass it rather than reimplementing it.

A wrapper may add robotorchan conventions such as:

- raw input / outcome provenance;
- `supports_mll`;
- `make_mll()`;
- a Mixed variant;
- small API normalization needed for consistency;
- explicit validation that protects robotorchan-specific semantics.

Do not copy substantial upstream implementation code merely to add these conveniences. Override only the smallest surface necessary.

## 4. Raw caller data must remain available

Where a model accepts training tensors, robotorchan should retain constructor-level provenance before preprocessing or BoTorch transforms are applied.

For ordinary supervised models this usually means:

- `raw_train_X`;
- `raw_train_Y`;
- `raw_train_Yvar`;
- `raw_data_names`;
- `raw_data`.

Rules:

- Store detached clones rather than aliases to caller-owned tensors.
- Preserve the caller-visible shape, dtype, and device semantics.
- Do not overwrite raw tensors with transformed, normalized, one-hot encoded, reshaped, broadcast, masked, or otherwise processed tensors.
- Avoid in-place mutation of caller tensors.
- Raw data represents constructor-level provenance; it is not required to mirror later `condition_on_observations()` or fantasy state.
- Use native BoTorch training attributes for the model's current conditioned state.

Model families with different data semantics should expose model-appropriate raw values instead of pretending to be ordinary supervised regression. Examples include `PairwiseGP`, `ModelListGP`, `HeterogeneousMTGP`, and `LatentKroneckerGP`.

## 5. Training-objective capability must be explicit

Public model wrappers expose a clear training capability contract.

- Exact GP wrappers normally set `supports_mll = True` and implement `make_mll()` using the appropriate marginal likelihood.
- Variational models may expose their natural ELBO through `make_mll()` when that is the established robotorchan convention.
- Fully Bayesian models that use NUTS set `supports_mll = False`; they must not pretend that an MLL-style fitting path exists.
- Specialized BoTorch fitting dispatch must remain usable.

`make_mll()` is a convenience for the model's native fitting objective, not a replacement for BoTorch fitting APIs.

## 6. Mixed models are separate public classes

Categorical-input support should normally be exposed through a separate Mixed class rather than adding `cat_dims` to the standard model.

Preferred naming follows BoTorch style:

```python
SingleTaskGP(...)
MixedSingleTaskGP(..., cat_dims=[1, 3])

MultiTaskGP(...)
MixedMultiTaskGP(..., cat_dims=[1, 3])
```

Use `MixedXxx`, not `XxxMixed`.

The non-mixed class should remain as close as possible to the upstream BoTorch constructor and semantics.

## 7. Mixed implementation policy: native categorical kernel first, internal one-hot second

Mixed models use the following decision rule.

```text
Mixed support required
        |
        v
Can the model safely use / replace its data covariance module?
        |
      yes ---------------- no / structurally unsafe
       |                         |
Native categorical kernel     Internal one-hot encoding
       |                         |
       +------------+------------+
                    |
               cat_dims API
```

### 7.1 Preferred path: native categorical kernel

Use the robotorchan mixed covariance infrastructure when the model's data covariance can be safely constructed or injected without changing upstream semantics.

The default native mixed covariance should preserve the established robotorchan form:

- continuous main effect;
- categorical main effect using `CategoricalKernel`;
- continuous × categorical interaction.

Structural columns such as task features or fidelity features must be excluded from the data covariance where appropriate.

Examples include the current Mixed variants for single-task, multitask, Kronecker multitask, variational, and multi-fidelity models.

### 7.2 Fallback path: internal one-hot encoding

If the upstream model evaluates covariance in a special backend or otherwise cannot safely use the normal GPyTorch mixed covariance, expose a Mixed class but perform categorical one-hot encoding internally.

The public API must remain in the original mixed feature space:

```python
model = MixedXxxGP(
    train_X=train_X,
    train_Y=train_Y,
    cat_dims=[1, 3],
)

posterior = model.posterior(raw_mixed_X)
```

Users should not need to manually one-hot encode training or prediction inputs.

Rules for internal one-hot Mixed models:

- Learn the category vocabulary from the training data unless the model explicitly supports a declared vocabulary.
- Store the category mapping on the model so the same encoding is used during training and prediction.
- Preserve the original mixed `raw_train_X`.
- Use the encoded tensor consistently throughout all internal fitting and posterior paths.
- Reject unseen categories explicitly unless a deliberate unknown-category policy is implemented and documented.
- Expose useful metadata such as normalized `cat_dims`, raw input dimension, encoded input dimension, and category values when practical.
- One-hot encoding is an implementation fallback, not a claim that the model uses a native categorical kernel.

Fully Bayesian SAAS is the canonical example: BoTorch evaluates the covariance inside the JAX/NumPyro NUTS path, so the Mixed wrapper uses internal one-hot encoding rather than a misleading post-hoc GPyTorch categorical kernel.

## 8. `cat_dims` conventions

Mixed models should use a consistent `cat_dims: list[int]` interface.

- Negative indices follow normal Python indexing semantics where supported by the shared helpers.
- Normalize and validate indices early.
- Reject duplicates.
- Reject out-of-range indices.
- Keep the normalized categorical dimensions available on the model, normally as an immutable tuple.
- Do not allow structural feature columns to also appear in `cat_dims`.

## 9. Structural feature columns must retain their semantics

Task, fidelity, context, hierarchy, and other structural features are not ordinary categorical design variables merely because their values may be integer-coded.

Examples:

- `task_feature` in long-format multitask models;
- `iteration_fidelity` and `data_fidelities` in multi-fidelity models;
- context / decomposition columns in contextual models;
- hierarchy indicators in hierarchical kernels.

Rules:

- Keep structural columns separate from `cat_dims` unless the mathematical model explicitly defines otherwise.
- Do not one-hot encode a task feature simply because a Mixed model uses internal one-hot encoding for design categories.
- Re-map structural feature indices carefully if preprocessing changes the input dimension.
- Add explicit overlap validation and tests.

## 10. Input transforms and preprocessing must not be double-applied

When robotorchan performs model-internal preprocessing, define clearly whether it happens before or after the upstream `InputTransform`.

- The same effective feature representation must be used during fitting and prediction.
- Do not encode training inputs one way and candidate / posterior inputs another way.
- Do not silently apply both a robotorchan preprocessing step and an upstream transform that duplicates the same operation.
- When input warping is used with a Mixed one-hot model, warp continuous raw features only unless there is a model-specific documented reason to do otherwise.
- Re-map indices after encoding rather than passing raw feature indices into an encoded feature space.

## 11. Keep standard and Mixed variants in the same model-family file when practical

Avoid unnecessary file proliferation.

Preferred organization:

```text
single_task.py
    SingleTaskGP
    MixedSingleTaskGP

multitask.py
    MultiTaskGP
    MixedMultiTaskGP
    KroneckerMultiTaskGP
    MixedKroneckerMultiTaskGP
```

Create a new file only when the model family is genuinely distinct or the existing file would become difficult to maintain.

Do not create generic files such as `mixed.py` merely to collect unrelated Mixed models.

Shared implementation details should go into a common helper only when they are genuinely reusable. Avoid both copy-paste duplication and premature framework-building.

## 12. Composition does not require synthetic Mixed container classes

Do not create a `MixedXxx` class solely because a container can hold Mixed child models.

For example, `ModelListGP` can combine standard and Mixed children directly, so a separate `MixedModelListGP` would add API surface without new semantics.

Add a Mixed class only when that class itself owns categorical-input behavior.

## 13. Public exports must be explicit and stable

Public models should be intentionally exported from `robotorchan.models`.

- Keep `__all__` explicit.
- Add new public classes to the public-model contract tests.
- Do not expose internal helper functions accidentally.
- Do not remove or rename existing public symbols casually.
- Prefer deprecation / migration for future breaking changes once the public API stabilizes.

## 14. Tests are part of the model contract

Every public model extension should have focused tests for the behavior it adds.

For wrappers, test as applicable:

- constructor compatibility with upstream BoTorch;
- public import / export;
- raw-data retention;
- `supports_mll` and `make_mll()` behavior;
- posterior tensor shapes and finite outputs;
- model-specific kernel / covariance structure;
- negative index handling;
- invalid / overlapping structural indices;
- custom kernel factory behavior;
- mixed categorical-only or continuous-only edge cases where meaningful;
- internal one-hot encoding metadata and unseen-category rejection;
- task / fidelity index preservation;
- serialization / dtype / device movement for registered buffers where relevant.

For Fully Bayesian models, remember that posterior tensors may retain an MCMC sample batch dimension. Tests should assert the actual upstream semantics rather than forcing exact-GP shapes.

## 15. CI success must be verified, not assumed

A change is not considered verified merely because code was pushed.

- Inspect the GitHub Actions run for the exact current head commit.
- Do not report CI success until the relevant workflow has completed successfully.
- When a job fails, inspect the failed job log and fix the actual failure rather than guessing.
- Distinguish implementation failures from incorrect test expectations, formatting failures, dependency issues, and workflow problems.
- Re-check the new workflow run after fixes.

Supported Python versions and optional-dependency jobs are part of the compatibility surface and should remain covered by CI.

## 16. Avoid hidden behavior and surprising convenience

Prefer explicit, inspectable behavior over magic.

- Do not silently reinterpret columns without a public argument or documented convention.
- Do not mutate global settings to make one model work.
- Do not silently switch mathematical models based on input values.
- Validate ambiguous or unsafe configurations early with clear errors.
- Internal convenience is welcome when it removes boilerplate without changing the user's conceptual model. Internal one-hot encoding for a class explicitly named `Mixed...` is an example of acceptable convenience.

## 17. Preserve native model semantics even when that limits uniformity

A uniform API is useful only when it remains truthful.

Examples:

- Fully Bayesian models use NUTS rather than `make_mll()`.
- `PairwiseGP` uses comparisons rather than pretending they are ordinary `train_Y` values.
- `ModelListGP` exposes grouped child training data rather than inventing one global tensor.
- `HeterogeneousMTGP` preserves per-task data structure.

Prefer a small, documented model-specific exception over an abstraction that misrepresents the underlying BoTorch model.

## 18. Clean-room development

robotorchan is implemented independently from public algorithms, papers, public documentation, and public APIs.

- Do not copy code from prior private or employer-owned projects.
- Do not use a prior internal library as a source implementation.
- Re-derive robotorchan-specific code from public behavior and the current robotorchan architecture.

## 19. Default implementation checklist

Before implementing a new model wrapper or extension, answer these questions:

1. Is there already an upstream BoTorch model that should be wrapped rather than reimplemented?
2. What constructor surface should remain identical to BoTorch?
3. What raw caller data must be retained?
4. Does the model use an exact MLL, ELBO, NUTS, or another fitting path?
5. Does a Mixed variant belong as a separate public class?
6. Can native categorical covariance be safely injected?
7. If not, can internal one-hot encoding preserve the same representation through fitting and prediction?
8. Are there structural feature columns that must be excluded from categorical handling?
9. Can the standard and Mixed implementations live in the same model-family file?
10. Which public exports and contract tests must change?
11. What model-specific behavior must be tested in addition to generic wrapper behavior?
12. Has CI for the exact final head commit completed successfully?

When these rules conflict, prioritize mathematical correctness and truthful BoTorch-compatible behavior over cosmetic API uniformity.

# Cross-capability model audit

This document records the current cross-capability surface of public models. It is an audit of
implemented semantics, not a request to create every Cartesian-product class.

## Status vocabulary

- **Runtime validated**: executable tests cover the relevant posterior or workflow.
- **Supported**: the implementation and public contract support the combination.
- **Structurally supported, unvalidated**: the implementation composes the axes, but the relevant
  end-to-end workflow does not yet have representative runtime coverage.
- **Intentionally unsupported**: the combination is outside the public contract by design.
- **Invalid**: the axes have incompatible semantics.

Runtime validation is a relation between a model and a workflow. It is therefore intentionally
kept separate from `ModelCapabilities`: a single model can be validated for posterior sampling
while remaining unvalidated for a particular acquisition or optimizer.

## Structural model-family matrix

| Family | Mixed | MultiTask | Mixed × MultiTask | Kronecker | Current status |
| --- | --- | --- | --- | --- | --- |
| Standard exact GP | yes | yes | yes | yes | supported |
| Reduced GP | yes | yes | yes | yes | supported |
| Neural / joint reduction | yes | yes | selected families | yes | supported where exported |
| SAAS fully Bayesian | yes | yes | yes | no dedicated variant | supported |
| Robust relevance pursuit | yes | yes | yes | no | supported |
| Heteroskedastic | yes | yes | yes | no | supported |
| Student-t / contaminated | yes | yes | yes | no | supported |
| Nonstationary | yes | yes | yes | no | supported |
| DeepGP | yes | yes | yes | no | supported |
| Infinite-width BNN | yes | yes | yes | no | supported |
| Spectral mixture | yes | yes | yes | no | supported |

Kronecker variants require aligned multi-output observations and are not aliases for long-format
MultiTask models. Their absence from a family is not automatically a missing implementation.

The expressive Mixed × MultiTask gap recorded by the earlier audit has been closed:
`MixedMultiTaskDeepGP`, `MixedInfiniteWidthBNNMultiTaskGP`, and
`MixedSpectralMixtureMultiTaskGP` are public implementations with executable posterior tests.

## Cross-capability workflow matrix

| Combination | Status | Evidence / boundary |
| --- | --- | --- |
| Mixed × MultiTask | Runtime validated | Reduced and expressive mixed multitask models have executable posterior tests; several robust families also have runtime tests. |
| Mixed × MultiFidelity | Runtime validated | `MixedSingleTaskMultiFidelityGP` has posterior, MC acquisition, KG, qMFKG, and mixed candidate-generation coverage. |
| Mixed × high-dimensional | Runtime validated | Mixed reduced models have posterior/MC coverage; reduced families have KG coverage. |
| Mixed × robust | Supported | Multiple public robust mixed models exist; runtime depth varies by robust family. |
| MultiTask × high-dimensional | Supported | Reduced, neural-reduced, SAAS, and expressive multitask implementations exist; validation depth varies by family. |
| MultiTask × robust | Supported | Relevance-pursuit, heteroskedastic, Student-t/contaminated, and nonstationary multitask implementations exist; runtime depth varies. |
| MultiFidelity × high-dimensional | Intentionally unsupported | No public combined model is declared; do not infer support from independent capabilities. |
| MultiFidelity × robust | Intentionally unsupported | No public combined model is declared; do not infer support from independent capabilities. |
| multi-output × acquisition | Runtime validated | Kronecker MultiTask GP has scalarized MC and log-EHVI/log-NEHVI runtime coverage. Broader model-family coverage is acquisition-specific. |
| ensemble × acquisition | Runtime validated in representative path | Tree empirical ensemble posterior is exercised with `IndexSampler` and MC acquisition. Gaussian ensemble semantics remain distinct. |
| Reduced GP × fantasy | Runtime validated | Continuous reduced models have qKG runtime coverage. |
| Mixed Reduced GP × fantasy | Runtime validated | Mixed reduced models have qKG runtime coverage. |

The table deliberately does not promote representative runtime evidence to a claim that every
model in a family is validated for every acquisition.

## Validation-depth gaps

The following are validation gaps, not known correctness defects:

- Mixed robust and MultiTask robust families do not all have the same posterior/acquisition depth.
  Nonstationary MultiTask has posterior plus MC acquisition coverage, while relevance-pursuit and
  heteroskedastic MultiTask coverage is narrower.
- MultiTask high-dimensional coverage is broad structurally, but acquisition E2E coverage is not
  exhaustive across reduction, neural reduction, SAAS, and expressive families.
- Ensemble validation currently proves a representative empirical-tree MC path, not arbitrary
  Active Learning acquisition semantics.
- Multi-fidelity combined with high-dimensional or robust modeling is not part of the current
  public model surface.

These gaps should be prioritized by workflow importance rather than represented as model-wide
booleans. The regression-specific validation plan and acceptance criteria are maintained in
[`regression_validation_matrix.md`](regression_validation_matrix.md).

## Capability-schema decision

`ModelCapabilities` should continue to describe structural and runtime-relevant model semantics,
such as input type, task type, inference type, posterior sampling type, and fantasy support.

Do **not** add a model-wide `runtime_validated` flag. Validation is workflow-specific and such a
flag would overstate coverage. If machine-readable validation metadata becomes necessary, add a
separate validation registry keyed by model and workflow / operation.

Mixed implementation strategy is also not added to the schema in this phase. Native categorical
kernels, one-hot transforms, and embeddings are implementation details unless an optimizer or
acquisition needs to branch on them. Add machine-readable categorical handling only when a real
consumer requires it.

## Public-contract rules

Every new public model must be reflected in `robotorchan.models.__all__`, the explicit public
model inventory, the relevant capability inventories, and `docs/model_coverage.json`.

A capability claim and a validation claim are different:

1. capability metadata states what the public model contract supports;
2. executable tests establish which workflows have been runtime validated;
3. acquisition compatibility must be justified by the model and acquisition requirements;
4. unsupported combinations must fail explicitly rather than being inferred from neighboring
   capabilities.

A missing cross-product class is not automatically a defect. Add one only when its statistical
semantics and public API are distinct and useful.

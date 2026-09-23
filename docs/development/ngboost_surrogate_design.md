# NGBoost probabilistic surrogate design

## Scope

Phase 14 defines the implementation contract for an optional NGBoost regression surrogate.
The goal is not to treat boosting stages as posterior members. NGBoost is valuable here because it
directly predicts a conditional probability distribution.

No public NGBoost class is added until the optional backend and posterior adapter satisfy this
contract.

## Statistical contract

For each candidate input `X`, the fitted NGBoost estimator returns a predictive distribution
`p(y | X)`. robotorchan should preserve that distributional meaning:

```text
raw torch X
    -> guarded CPU / numpy conversion
    -> NGBoost pred_dist
    -> distribution-aware BoTorch Posterior
    -> rsample / sample
    -> MC acquisition or regression active learning
```

Individual boosting stages are **not** posterior samples. NGBoost must therefore not reuse the
complete-model tree `EnsemblePosterior` contract used by Random Forest, Extra Trees, or bootstrap
boosting.

## Initial public scope

The first implementation should deliberately support:

- regression only;
- one output;
- numeric continuous input;
- explicit `fit()`;
- `raw_train_X` and `raw_train_Y` preservation;
- `supports_mll = False`;
- no candidate-input gradients;
- Gaussian NGBoost predictive distribution first;
- MC acquisition functions;
- gradient-free acquisition search;
- regression active-learning acquisitions whose semantics require predictive samples or predictive
  variance.

Mixed inputs, multi-output, classification, and arbitrary NGBoost distributions are later gates.

## Optional dependency

NGBoost must remain optional. Importing `robotorchan` without `ngboost` installed must work.
Constructing the NGBoost adapter without the dependency should raise an actionable `ImportError`.

A dedicated optional dependency group is preferred over adding NGBoost to the core install.

## Posterior adapter

The posterior must satisfy the BoTorch `Posterior` shape contract while retaining the configured
predictive law. For the first Gaussian implementation, it should expose at least:

- `mean`;
- `variance`;
- `rsample(sample_shape)`;
- output dtype/device matching candidate `X`;
- batch / q / m shape behavior required by MC acquisitions.

Sampling may originate from NGBoost distribution parameters and be performed with torch when that
preserves the same law. Candidate-input autograd is still unsupported because backend prediction
passes through CPU/numpy.

The implementation must not advertise a generic Gaussian posterior capability for future non-Gaussian
NGBoost distributions. Capability metadata should describe the implemented adapter, not NGBoost as a
library in the abstract.

## Uncertainty semantics

NGBoost's conditional predictive distribution represents total predictive uncertainty under the
fitted distributional model. That is enough for several useful workflows:

| Workflow | Initial support | Reason |
| --- | --- | --- |
| MC Bayesian optimization | yes | sampleable predictive distribution |
| predictive variance AL | yes | variance is directly available |
| predictive entropy AL | distribution-dependent | valid when entropy is defined by adapter |
| interval / tail sampling | distribution-dependent | valid when predictive law supports it |
| BALD | no | requires epistemic / aleatoric decomposition |
| analytic GP acquisitions | no general claim | GP posterior assumptions do not follow from NGBoost |

In particular, a Gaussian NGBoost predictive distribution must not be described as a Gaussian
Process posterior.

## BALD boundary

Plain NGBoost does not automatically expose the model-posterior decomposition required by BALD.
Sampling from `p(y | X)` alone mixes uncertainty sources and is insufficient to claim mutual
information between predictions and model parameters.

A future bootstrap-NGBoost ensemble could provide an explicit epistemic outer ensemble with
distributional aleatoric uncertainty inside each member. That is a separate model contract and
should not be silently folded into the first adapter.

## Proposed API

The initial class should be small:

```python
model = NGBoostSurrogate(
    train_X,
    train_Y,
    random_state=0,
    **ngboost_kwargs,
)
model.fit()
posterior = model.posterior(X)
samples = posterior.rsample(torch.Size([128]))
```

Backend-specific estimator customization can be exposed only where it does not create a second,
incompatible robotorchan API.

## Runtime acceptance criteria

Before public export, tests must cover:

1. robotorchan imports without NGBoost installed;
2. missing dependency raises an actionable error only when the adapter is used;
3. raw training tensors are retained exactly;
4. one-output and numeric-input validation;
5. deterministic seeding where NGBoost supports it;
6. posterior mean and variance shapes;
7. finite posterior samples with BoTorch-compatible shape;
8. candidate dtype/device restoration;
9. `make_mll()` remains unsupported;
10. observation-noise arguments are rejected unless explicitly implemented;
11. representative MC acquisition evaluation;
12. representative gradient-free candidate selection;
13. predictive-variance regression AL;
14. no BALD capability claim.

## Implementation sequence

Phase 15 should implement the smallest Gaussian NGBoost adapter and posterior contract. It should not
add mixed, multi-output, classification, or BALD support at the same time.

After the base adapter is executable:

1. validate MC BO;
2. validate regression AL based on predictive variance;
3. benchmark against GP and empirical tree surrogates;
4. decide whether non-Gaussian NGBoost distributions justify a generic distribution-posterior
   abstraction;
5. consider bootstrap-NGBoost only if epistemic decomposition is practically required.

This keeps the non-GP path BoTorch-compatible without overstating the uncertainty semantics.

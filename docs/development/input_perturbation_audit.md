# Input perturbation compatibility audit

Status: Phase 1 audit  
Baseline: `main@102281ca5e464e8070b6a301badd4df32e6f779f`

## Scope

This audit separates three concepts:

1. uncertain-input surrogate modelling,
2. observation / process robustness,
3. decision-time input perturbation of a nominal candidate followed by scenario-risk aggregation.

Phase 1 concerns item 3. Existing `UncertainInputSingleTaskGP` and
`MixedUncertainInputSingleTaskGP` do not by themselves establish that contract.

## Baseline findings

The public model coverage contains 142 model exports.

Repository-wide inspection found:

- no use of BoTorch `InputPerturbation`,
- no model capability field for decision-time perturbation,
- existing risk aggregation for expectation, mean-variance, worst-case, VaR,
  CVaR, and signal-to-noise ratio,
- posterior-sampling and fantasize capability metadata that are useful
  prerequisites but are not sufficient evidence of perturbation compatibility.

Therefore no public model is currently **certified by robotorchan** as
decision-time input-perturbation compatible. This does not mean every model is
incompatible; it means runtime evidence is not yet represented by the library.

## Compatibility classes

| Family | Phase-1 classification | Required validation |
| --- | --- | --- |
| Exact single-task GP | candidate | perturb -> posterior -> MC objective -> acquisition -> optimize |
| Variational / fully Bayesian GP | candidate | posterior sampling and acquisition shape semantics |
| Mixed input GP | conditional | perturb continuous dimensions only |
| MultiTask GP | conditional | task feature must never be perturbed |
| Kronecker multitask GP | conditional | preserve task/output structure |
| Multi-fidelity GP | conditional | fidelity dimensions are protected by default |
| Mixed + multitask / fidelity | conditional | protect the union of structural dimensions |
| PCA / PLS / random projection | conditional | perturb raw design space before reduction |
| AE / VAE / joint encoder | conditional | validate raw-space perturbation and transform path |
| Robust observation models | candidate/conditional | keep observation robustness distinct |
| Structured / contextual / hierarchical | conditional | protect context, hierarchy and discrete structure |
| HOGP / latent Kronecker | conditional | dedicated structured-posterior E2E tests |
| DeepGP / infinite-width BNN / spectral mixture | candidate/conditional | sampler and acquisition runtime validation |
| ModelListGP | conditional | every component and objective mapping must work |
| PairwiseGP | unsupported pending design | preference likelihood is outside the regression risk-objective contract |
| deterministic tree / boosting surrogates | unsupported pending design | require a separate scenario-evaluation adapter |
| NGBoostSurrogate | conditional prototype | probabilistic scenario aggregation is plausible, but this is not a BoTorch InputPerturbation contract |
| uncertain-input models | separate mechanism | require the same E2E contract before certification |

## Protected-dimension contract

Perturbations apply to design/environmental continuous coordinates, not
structural coordinates. Protected dimensions include categorical, task,
fidelity, context, hierarchy, and other discrete structural coordinates.

For reduced models the default semantic contract should be:

```text
nominal raw X
  -> perturb allowed raw dimensions
  -> model-owned reduction / representation
  -> posterior
  -> scenario risk aggregation
```

This avoids treating arbitrary latent-space offsets as physical uncertainty.

## Phase 1 conclusion

The repository has useful prerequisites, especially posterior capability
metadata and risk aggregators, but lacks an explicit runtime-backed
input-perturbation contract.

Phase 2 should add capability semantics before broad implementation. The
capability must distinguish certified support, conditional support requiring
protected dimensions, and unsupported/separate mechanisms.

Certification must be based on runtime tests rather than constructor
signatures or inheritance.

## Planned certification path

```text
raw candidate
 -> perturb selected dimensions
 -> posterior
 -> posterior samples
 -> scenario-aware risk objective
 -> MC acquisition
 -> candidate optimization
```

Mixed, multitask, multi-fidelity, contextual and hierarchical families must
also prove that protected coordinates remain unchanged.

Phase 1 intentionally adds no model implementation and makes no unsupported
compatibility claims.


## Phase 2 capability contract

`ModelCapabilities.input_perturbation` now records the audit/certification state:

- `unverified`: plausible path, but no E2E certification yet,
- `conditional`: structural/protected-dimension or adapter constraints apply,
- `unsupported`: the generic regression perturbation contract does not apply,
- `separate_mechanism`: the model addresses uncertain inputs through a different modelling contract,
- `supported`: reserved for models that pass the runtime certification path.

Phase 2 intentionally assigns no model to `supported`. Later phases must promote models only after runtime evidence.

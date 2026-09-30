# Batch, asynchronous, and fantasization compatibility

This page records the executable compatibility contract established by the batch / asynchronous /
fantasization audit. It describes runtime evidence rather than adding a parallel scheduler API.

## Responsibility boundaries

- Models own posterior sampling, conditioning, and `fantasize()` when their model contract
  supports it.
- Acquisitions own pending-point semantics through BoTorch `set_X_pending()`.
- Optimizers own joint q-batches, native sequential optimization, and search-space constraints.
- Asynchronous BO is the composition of completed training observations and unresolved
  `X_pending`; robotorchan does not maintain a separate async scheduler state.
- Knowledge-gradient and multi-step lookahead fantasy variables are acquisition-internal and are
  not interchangeable with real unresolved `X_pending` evaluations.

## Runtime compatibility

| Area | q-batch | X_pending / async | Fantasize | Notes |
| --- | --- | --- | --- | --- |
| Standard exact GP | Supported | Supported | Supported | Full fantasy-to-acquisition q-batch E2E |
| MultiTaskGP | Supported | Supported through compatible acquisitions | Supported | Fantasy inputs must identify tasks |
| Kronecker MultiTask GP | Supported | Supported through compatible acquisitions | Unsupported | GPyTorch fantasy prediction-cache contract is not satisfied by the specialized posterior path |
| Mixed GP | Supported | Supported | Model-dependent | Native mixed optimizer; inter-point nonlinear mixed constraints remain unsupported |
| Reduced / high-dimensional exact GP | Supported | Supported | Supported where advertised | Candidate and pending coordinates remain in original input space |
| Multi-fidelity GP | Supported | Supported | Supported where advertised | Pending fidelity is real evaluation context; MF-KG fantasies are separate |
| Multi-objective GP | Supported | Supported | Model-dependent | qLogEHVI / qLogNEHVI integration tested |
| Input-perturbation robust BO | Supported | Supported | Model-dependent | Input scenarios, posterior MC samples, and fantasies are distinct axes |
| Empirical tree ensembles | Supported with compatible sampler / optimizer | Acquisition-dependent | Unsupported | Do not impose ExactGP fantasy semantics |
| NGBoost | Posterior-path dependent | Acquisition-dependent | Unsupported | Optional non-GP dependency |
| qKG / lookahead | One-shot semantics | Separate from internal fantasies | Required by acquisition | Augmented rows are not all real candidates |

## Explicit limitations

Kronecker models intentionally advertise `supports_fantasize=False`. A private prediction-cache
workaround would couple robotorchan to GPyTorch internals and is not part of the public contract.

Mixed one-shot qKG / qMFKG candidate generation remains correctness-first and `q=1`. The exact
categorical enumeration scales over the full augmented batch, so blindly allowing larger q would
create a combinatorial path without a justified scalable algorithm.

A dedicated `supports_X_pending` model capability is intentionally not introduced. Pending-point
support is primarily an acquisition contract and depends on the selected acquisition rather than
being an intrinsic model property.

## Shape contract

The external candidate dimension `q`, fantasy-model batch dimensions, input-perturbation
scenario dimensions, and one-shot auxiliary rows are independent. Tests must not infer one from
another. In particular, a fantasy batch of size two does not imply `q=2`, and one-shot augmented
rows must be removed through the acquisition's candidate-extraction semantics before returning
real evaluation candidates.


## Initialization contract

Acquisition initialization remains a candidate-search concern. Real unresolved `X_pending` points
are acquisition context and are not appended to `batch_initial_conditions`. For ordinary q-batch
optimization, explicit initial conditions therefore keep shape `num_restarts x q x d` regardless
of the number of pending evaluations.

Likewise, a model fantasy batch is a model batch dimension, not an optimizer restart or candidate
dimension. Standard initialization still uses `num_restarts x q x d`; BoTorch broadcasts the
acquisition evaluation over compatible fantasy-model batch dimensions. One-shot acquisition
fantasy rows are different: those are acquisition decision variables and follow the augmented-q
initialization contract documented for KG and multi-step lookahead.

## Extension rule

Add new Batch / Async / Fantasization functionality only when BoTorch primitives cannot express
the required workflow, the missing responsibility is clear, shapes and gradients are defined,
and executable tests justify the maintenance cost.

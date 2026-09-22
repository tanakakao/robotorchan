# Standard single-objective acquisition functions

robotorchan uses BoTorch's native acquisition classes directly for standard
single-objective Bayesian optimization. It does not duplicate or re-export these classes.

## Recommended defaults

For improvement-based Bayesian optimization, prefer the numerically robust log-space
implementations supplied by BoTorch:

- `LogExpectedImprovement` for analytic q=1 expected improvement;
- `qLogExpectedImprovement` for Monte Carlo batch expected improvement;
- `qLogNoisyExpectedImprovement` when observations are noisy.

BoTorch warns that legacy `qExpectedImprovement` has known numerical issues and recommends
`qLogExpectedImprovement` instead. The same design principle applies to noisy improvement
methods. Legacy EI classes may still be used directly from BoTorch when a reproduction or
benchmark explicitly requires them, but robotorchan does not recommend them as defaults.

Other standard choices remain native BoTorch APIs:

- `LogProbabilityOfImprovement` for probability of improvement;
- `UpperConfidenceBound` and `qUpperConfidenceBound` for confidence-bound exploration;
- `qSimpleRegret` for posterior-utility / simple-regret style selection.

## Example

```python
from botorch.acquisition.logei import qLogExpectedImprovement

acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())
```

The returned object is a BoTorch acquisition function and can be passed directly to
`botorch.optim.optimize_acqf`.

## Model compatibility

Standard GP-like robotorchan models expose BoTorch-compatible posteriors, so acquisition
compatibility is a property of the model/posterior contract rather than a robotorchan
wrapper. The integration suite includes a test with `SingleTaskGP` for analytic and Monte
Carlo acquisitions.

Non-GP empirical ensemble surrogates are different: use the validation helpers documented
in the acquisition architecture because analytic Gaussian acquisitions are not valid for
those empirical posteriors.

Multi-objective, multi-fidelity, information-theoretic, lookahead, active-learning, and
level-set acquisitions are covered by dedicated guides.


For the broader theory and method relationships, see
[Acquisition theory](../theory/acquisition/12_selection_guide.md).

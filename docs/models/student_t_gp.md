# Student-t observation GP models

Student-t likelihood models are **implemented** robust surrogates for
heavy-tailed observation residuals. They are distinct from sparse-outlier
relevance pursuit and input-dependent Gaussian observation noise.

## Statistical model

```text
f(x) ~ GP
y | f(x) ~ StudentT(df=nu, loc=f(x), scale=sigma)
```

The public constructors require `df > 2` so the observation variance
exists. These models use variational inference with a Student-t
likelihood, not exact Gaussian marginal likelihood fitting.

## Public model families

- `StudentTSingleTaskGP`: continuous single-task scalar regression.
- `MixedStudentTSingleTaskGP`: continuous and categorical inputs
  with `cat_dims` and a mixed covariance.
- `StudentTMultiTaskGP`: long-format multi-task data using
  `task_feature` and task covariance.
- `MixedStudentTMultiTaskGP`: long-format multi-task data with
  mixed continuous/categorical data covariance.

Task features and categorical data features are separate structural
roles. Choose the matching model rather than interpreting task IDs or
category codes as continuous measurements.

## Training and posterior contract

```python
import torch
from robotorchan.models import StudentTSingleTaskGP

train_X = torch.rand(20, 2, dtype=torch.double)
train_Y = torch.sin(train_X[:, :1])

model = StudentTSingleTaskGP(
    train_X,
    train_Y,
    num_inducing=16,
    df=4.0,
)
loss = model.training_loss()
posterior = model.posterior(train_X[:3])
```

`training_loss()` combines the Student-t expected log likelihood and
variational KL term. Optimize the differentiable loss with a suitable
PyTorch optimizer. `supports_mll = False`, and `make_mll()` raises
rather than returning an exact Gaussian MLL. The BoTorch-compatible
`posterior()` describes the **latent response**, not a Student-t
predictive observation distribution.

The implementation preserves raw training inputs and outputs, and
clears variational caches when loading a state dictionary.

## Scope and interpretation

The heavy-tailed likelihood is a modeling assumption, not a guarantee
of outlier resistance in every dataset. Verify calibration and
predictive performance against Gaussian and other robust baselines
on held-out data. Input perturbation and CVaR/worst-case acquisition
risk remain separate optimization concerns.

See [Robust and noise models](robust_noise.md) for model selection.

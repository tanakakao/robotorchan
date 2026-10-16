# 7. Multi-task and Multi-output Gaussian Processes

## 7.1 Definitions

A **multi-output** model jointly describes a vector-valued
function \(f(x)=(f_1(x),\ldots,f_T(x))\).
A **multitask** model relates observations associated with
different tasks, which may be observed at different inputs.
A **multiobjective** optimization problem specifies several
objectives to optimize; it does not prescribe a particular
surrogate covariance.

The defining question for a multitask GP is whether and how
observations from one task inform predictions for another.

## 7.2 Coregionalization

For input \(x\in\mathcal X\) and task \(t\), an intrinsic
coregionalization model (ICM) uses

$$
\operatorname{Cov}\bigl(f_t(x),f_{t'}(x')\bigr)
=
k_X(x,x')B_{tt'},
$$

where \(k_X\) is a valid input kernel and
\(B\succeq0\) is a task covariance matrix.

A low-rank parameterization is

$$
B=WW^\top+\operatorname{diag}(v),
\qquad v_t\ge0.
$$

The rank of \(W\) controls shared task structure; the
nonnegative diagonal term allows task-specific variation.
The covariance \(B_{tt'}\) can be positive, negative, or
zero. Its sign does not establish a causal relationship.

ICM is separable: all tasks share the same input kernel
shape up to task covariance. More expressive linear models
of coregionalization use sums

$$
k\big((x,t),(x',t')\big)
=
\sum_{q=1}^{Q} k_q(x,x')B^{(q)}_{tt'},
$$

which permit different shared latent processes to have
different input lengthscales.

## 7.3 Observation designs

### Long-format observations

An observation is a triple \((x_i,t_i,y_i)\).
Different tasks may be observed at different \(x_i\), and
some task-input combinations may be missing.

The covariance matrix is formed entrywise:

$$
K_{ij}=k_X(x_i,x_j)B_{t_i t_j}.
$$

No complete Cartesian observation grid is required.

### Complete block design

If all \(T\) tasks are observed at the same \(n\) input
locations and the observations are ordered consistently,
the latent covariance has Kronecker structure:

$$
K_{\mathrm{latent}}=B\otimes K_X.
$$

This structure can enable computational savings through
structured linear algebra. The *full observation covariance*
also contains noise, which may break simple Kronecker
factorization unless its structure is compatible.

A complete block design is sufficient for this matrix
form, but it is not required for the underlying separable
kernel to be mathematically valid.

## 7.4 Independent versus correlated outputs

Independent GP models correspond to a block-diagonal
cross-output covariance. They do not share information
across outputs, although each can use its own kernel,
likelihood, and training data.

A correlated multitask model can improve sample efficiency
when tasks are related. It can also suffer **negative
transfer** if cross-task correlations are estimated poorly
or the separable assumptions are incorrect.

The distinction between independent and correlated outputs
is statistical, not merely a choice of container type.

## 7.5 Missing observations and noise

When tasks have unequal numbers of observations, long-format
modeling naturally represents observed pairs without
inventing unmeasured outputs.

For noisy observations,

$$
y_i=f_{t_i}(x_i)+\epsilon_i,
\qquad
\operatorname{Cov}(\epsilon_i,\epsilon_j)
=
\delta_{ij}\sigma^2_{t_i}(x_i)
$$

under independent observation noise.
Known and learned noise variances imply different
inference assumptions. Standardizing outcomes by task
changes parameter scales and must be inverted when
interpreting predictions in physical units.

## 7.6 Posterior covariance and downstream objectives

A multi-output posterior contains a mean vector and a
cross-output covariance. For a linear scalarization
\(g(x)=w^\top f(x)\),

$$
\mathbb E[g(x)]=w^\top\mu(x),
\qquad
\operatorname{Var}[g(x)]
=
w^\top\Sigma(x)w.
$$

Ignoring off-diagonal covariance terms can misstate
uncertainty in scalarized objectives or joint constraints.

A multiobjective acquisition instead operates on the
selected physical objective vector and its predictive
distribution. Task identity, output identity, and
optimization objective semantics are separate concepts.

## 7.7 Multitask versus multi-fidelity

Multitask modeling represents relationships among tasks
or data sources. Multi-fidelity modeling additionally
assigns an accuracy/cost or resolution interpretation
to fidelity levels. A multi-fidelity surrogate may use
multitask kernels, but a generic task index does not
encode fidelity ordering or evaluation cost.

Similarly, task indices should not be treated as
ordinary numeric distances without an explicit model
justifying that geometry.

## 7.8 Assumptions and diagnostics

- Validate transfer performance against independent
  per-task baselines.
- Check whether separability and shared input
  lengthscales are plausible.
- Assess sensitivity to task covariance rank and
  strongly imbalanced task datasets.
- Distinguish latent covariance from observation noise.
- A strong estimated task correlation does not
  establish a causal relationship.

## References

1. Bonilla, E. V., Chai, K. M. A., & Williams, C. K. I. (2008).
   Multi-task Gaussian Process Prediction. *NeurIPS*.
2. Álvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012).
   Kernels for Vector-Valued Functions: A Review.
   *Foundations and Trends in Machine Learning*, 4(3), 195–266.
3. Swersky, K., Snoek, J., & Adams, R. P. (2013).
   Multi-task Bayesian Optimization. *NeurIPS*.

## Related documents

- [Gaussian Process](02_gaussian_process.md)
- [Kernel](03_kernel.md)
- [Multi-Fidelity](06_multi_fidelity.md)
- [Hierarchical / Contextual GP](12_hierarchical_contextual_gp.md)
- [Model documentation](../models/README.md)

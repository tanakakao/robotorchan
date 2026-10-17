# Nonstationary Gaussian Processes

## 1. Stationarity and local smoothness

A covariance function is stationary when
\(k(x,x')=\kappa(x-x')\). Isotropy is a stronger restriction:
covariance depends only on a distance such as \(\|x-x'\|\).
An automatic relevance determination (ARD) RBF kernel may be
anisotropic yet stationary:

$$
k_{\mathrm{RBF}}(x,x')
=\sigma_f^2\exp\left(
-\frac12\sum_{d=1}^D\frac{(x_d-x_d')^2}{\ell_d^2}
\right).
$$

Global lengthscales \(\ell_d\) cannot represent a correlation length
that changes with location. Nonstationary covariance models allow local
properties of the latent function to vary over the input domain.

This differs from heteroskedasticity, where the *observation noise*
variance changes with \(x\), and uncertain-input models, where the
location of an observation is itself uncertain.

## 2. Gibbs covariance with input-dependent lengthscales

Let \(\ell_d(x)>0\) be the local lengthscale of coordinate \(d\).
A Gibbs-type covariance can be written as

$$
k(x,x')=\sigma_f^2
\prod_{d=1}^{D}
\left(
\frac{2\ell_d(x)\ell_d(x')}
{\ell_d(x)^2+\ell_d(x')^2}
\right)^{1/2}
\exp\left(
-\sum_{d=1}^{D}
\frac{(x_d-x_d')^2}
{\ell_d(x)^2+\ell_d(x')^2}
\right).
$$

This construction is positive semidefinite for positive local
lengthscale functions. If each \(\ell_d(x)\equiv\ell_d\) is constant,
the prefactor becomes one and the exponent reduces to the stationary
ARD RBF kernel.

A small local lengthscale allows shorter-range correlation; it does
**not** by itself establish a physical phase boundary or causal
regime transition.

## 3. Parameterizing local lengthscales

One finite-dimensional model is

$$
\ell_d(x)
=
\operatorname{softplus}\!\left(
b_d+\sum_{j=1}^{D}W_{dj}x_j
\right)+\ell_{\min,d},
\qquad \ell_{\min,d}>0.
$$

Here \(W\) controls how each local lengthscale responds to other input
coordinates. The softplus and positive floor enforce positivity.
This is a restricted nonstationary model, not a universal
representation of arbitrary smoothness fields. More flexible
alternatives place stochastic processes on local kernel parameters
or construct nonstationary covariances through input warping.

The model's additional degrees of freedom can be difficult to
identify with small datasets. Validation against a stationary
baseline is therefore important.

## 4. Structured input spaces

For mixed continuous and categorical inputs, a nonstationary
continuous covariance may be combined with a valid categorical
kernel through sums or products. Category identifiers are not
automatically meaningful Euclidean coordinates.

For multitask data, a separable construction is

$$
k\big((x,t),(x',t')\big)
=
k_{\mathrm{data}}(x,x')\,k_{\mathrm{task}}(t,t').
$$

The data covariance may be nonstationary while the task covariance
captures relationships among discrete tasks. With a complete
Cartesian observation design, its covariance matrix has a Kronecker
structure, up to the ordering of observations:

$$
K=K_{\mathrm{task}}\otimes K_{\mathrm{data}}.
$$

This factorization is an assumption, not a universal property of
multitask nonstationary GPs. Other models allow task-specific local
lengthscale fields and nonseparable cross-task interactions.

## 5. Interpretation and limitations

- Local lengthscales describe covariance geometry, not posterior
  uncertainty or a calibrated physical boundary.
- Spatial variation in observation scatter is a separate
  heteroskedastic likelihood problem.
- Apparent nonstationarity can also result from an inadequate mean
  function, missing covariates, outliers, or known categorical regimes.
- Nonstationary kernels may reduce underfitting but increase
  estimation variance and numerical complexity.
- Held-out predictive performance and posterior predictive checks
  should guide model comparison.

## 6. 非定常性の診断と推定上の注意点

非定常カーネルは場所ごとに異なる相関距離を表現できますが、
観測点が疎な領域では局所長さ尺度 $\ell_d(x)$ を十分に識別できないことがあります。
局所的に長さ尺度が短く推定されたことを、直ちに物理的な相転移や
異常発生の証拠と解釈することは適切ではありません。

また、入力に依存する観測ノイズと入力に依存する潜在関数の滑らかさは
別の統計的要因です。反復測定があれば観測ノイズの変化を推定しやすくなりますが、
単一観測のみでは両者の区別が難しい場合があります。

評価では、定常カーネルを基準モデルとして、予測対数尤度、
校正、領域別の残差、および未観測領域での不確実性を比較します。
追加の柔軟性による改善が再現されるかを確認することが重要です。

## References

1. Gibbs, M. N. (1997). *Bayesian Gaussian Processes for Regression
   and Classification*. PhD thesis, University of Cambridge.
2. Paciorek, C. J., & Schervish, M. J. (2006).
   Spatial modelling using a new class of nonstationary covariance
   functions. *Environmetrics*, 17(5), 483–506.
3. Rasmussen, C. E., & Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## Related documents

- [Gaussian Process](02_gaussian_process.md)
- [Kernel](03_kernel.md)
- [Heteroskedastic / Replicate Noise](15_heteroskedastic_noise.md)
- [Uncertain-input Gaussian Process](16_uncertain_input_gp.md)
- [MultiTask / MultiOutput](07_multitask_multioutput.md)
- [Library-specific nonstationary GP models](../models/nonstationary_gp.md)

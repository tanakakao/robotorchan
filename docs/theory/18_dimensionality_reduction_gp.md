# Dimensionality Reduction for Gaussian Process Models

## 1. Modeling dimension versus search dimension

Let \(x\in\mathbb R^D\) denote the original input and
\(z=T(x)\in\mathbb R^d\), with \(d<D\), a reduced representation.
A reduced-input surrogate assumes

$$
f(x)\approx g(T(x)),\qquad g\sim\mathcal{GP}(m_z,k_z).
$$

This changes the *modeling dimension*. It does not automatically
reduce the dimension of the acquisition optimization problem when
candidates must still be chosen in the original space.

A separate search embedding instead optimizes a low-dimensional
decision variable \(z\) and maps it to a feasible original input
\(x=S(z)\). The two transformations need not be inverses.

## 2. Principal component analysis

For centered observations \(X_c\in\mathbb R^{n\times D}\),
PCA selects orthonormal directions maximizing sample variance.
If \(V_d\in\mathbb R^{D\times d}\) contains the leading right
singular vectors of \(X_c\), the representation is

$$
z=V_d^\top(x-\bar x).
$$

PCA is unsupervised: a low-variance direction can still be highly
predictive of the response. PCA component count is therefore a
model-selection parameter, not solely a variance-explained threshold.

## 3. Partial least squares

Partial least squares (PLS) constructs latent directions using
input-output relationships. A basic PLS objective selects directions
that favor covariance between projected predictors and responses.
Unlike PCA, PLS is supervised and can retain predictive directions
with relatively low input variance.

Because PLS uses labels, all fitting of its projection—including
component selection—must occur inside the training partition of
cross-validation. Otherwise evaluation suffers from target leakage.

## 4. Random projection

A random linear map \(R\in\mathbb R^{d\times D}\) produces

$$
z=Rx.
$$

Suitable random constructions approximately preserve pairwise
distances on a *finite set* with high probability when the
projection dimension is sufficiently large. This is the setting
of Johnson–Lindenstrauss-type guarantees; such guarantees do not
imply that a particular low-dimensional projection preserves the
response function or its optimum.

## 5. Induced covariance and identifiability

For a fixed transformation \(T\), a reduced-input GP induces

$$
k_x(x,x')=k_z(T(x),T(x')).
$$

This is positive semidefinite whenever \(k_z\) is a valid kernel.
However, if \(T(x)=T(x')\), the surrogate cannot distinguish
\(x\) from \(x'\) through this covariance. Any variation in \(f\)
along directions discarded by \(T\) is necessarily misspecified
unless separately modeled.

Learning \(T\) jointly with the GP introduces additional
nonidentifiability: multiple representations and kernel parameters
may define the same predictive function.

## 6. Output reduction and uncertainty restoration

Suppose \(y\in\mathbb R^M\) and a linear output representation is

$$
y\approx \mu_y+A z_y,\qquad A\in\mathbb R^{M\times r}.
$$

If the reduced predictive distribution has mean \(\mu_z\) and
covariance \(\Sigma_z\), the reconstructed output moments are

$$
\mathbb E[y]\approx\mu_y+A\mu_z,
\qquad
\operatorname{Cov}(y)\approx A\Sigma_zA^\top.
$$

If an independent residual model with covariance \(\Sigma_\epsilon\)
is included, it adds to the reconstructed covariance. Without that
term, uncertainty in discarded output directions is omitted.
Nonlinear decoders generally require sampling or approximate
moment propagation.

A downstream multiobjective acquisition must use the intended
original-output semantics, not silently treat latent coordinates
as physical objectives.

## 7. Mixed, multitask, and structured inputs

Categorical identifiers, task indices, fidelity levels, and
context labels have distinct statistical meanings. An ordinary
Euclidean projection of their numeric encodings can introduce
spurious distances.

One possible mixed-input construction reduces continuous design
variables only:

$$
T(x_{\mathrm{cont}},x_{\mathrm{cat}})
=
\big(T_c(x_{\mathrm{cont}}),x_{\mathrm{cat}}\big).
$$

The resulting kernel can combine continuous reduced-space and
categorical covariances. Task and fidelity coordinates may be
modeled through their own covariance structures rather than
being absorbed into an undifferentiated PCA projection.

## 8. Statistical and optimization limitations

- The reduction is an inductive bias and may discard
  response-relevant directions.
- Validation must refit supervised and unsupervised reductions
  within each training fold to avoid information leakage.
- A reduced modeling space does not guarantee a reduced feasible
  search space.
- Noninvertible projections make original-space candidate
  reconstruction ambiguous.
- Output reconstruction can understate predictive uncertainty
  when discarded directions are ignored.
- Projection dimension and surrogate hyperparameters should be
  assessed together.

## References

1. Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis:
   A review and recent developments. *Philosophical Transactions of
   the Royal Society A*, 374.
2. Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression:
   A basic tool of chemometrics. *Chemometrics and Intelligent
   Laboratory Systems*, 58(2), 109–130.
3. Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of
   Lipschitz mappings into a Hilbert space. *Contemporary Mathematics*, 26.
4. Constantine, P. G., Dow, E., & Wang, Q. (2014). Active subspace
   methods in theory and practice: Applications to kriging surfaces.
   *SIAM Journal on Scientific Computing*, 36.
5. Rasmussen, C. E., & Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## Related documents

- [Gaussian Process](02_gaussian_process.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Neural reduction GP](19_neural_reduction_gp.md)
- [High-dimensional search](20_high_dimensional_search.md)
- [MultiTask / MultiOutput](07_multitask_multioutput.md)
- [Input-reduction model guide](../models/high_dimensional_inputs.md)
- [Output-reduction model guide](../models/high_dimensional_outputs.md)

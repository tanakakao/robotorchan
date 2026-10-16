# Hierarchical, Heterogeneous Multitask, and Contextual Gaussian Processes

## 1. Distinct forms of structure

Several GP constructions exploit information beyond a flat Euclidean
design space. The distinctions are statistical:

| Structure | Mathematical issue |
| --- | --- |
| Hierarchical design | A parent choice activates or deactivates child coordinates. |
| Heterogeneous multitask | Different tasks observe different subsets of covariates. |
| Contextual decomposition | An output decomposes into contributions associated with contexts. |
| Context embeddings | Similarity among contexts is learned through latent descriptors. |

These assumptions should not be conflated. A categorical design variable
does not necessarily induce a hierarchy, and a task label is not
automatically a continuous coordinate.

## 2. Hierarchical search spaces

Let \(b\) be a branch variable and \(x=(x_1,\ldots,x_D)\) the
candidate coordinates. Define an activity indicator

$$
a_j(b)\in\{0,1\},\qquad
A(b)=\{j:a_j(b)=1\}.
$$

A hierarchical response \(f(b,x)\) is invariant to inactive coordinates:
if \(x_j\) and \(x_j'\) differ only where \(a_j(b)=0\), then

$$
f(b,x)=f(b,x').
$$

For example, one manufacturing route may use temperature while another
uses pressure. An arbitrary placeholder in an inactive coordinate
must not change predictions.

### 2.1 Covariance constructions

A valid hierarchical covariance must be positive semidefinite and
respect inactive-coordinate invariance. One simple, explicit
construction is a sum of shared and branch-specific kernels:

$$
k((b,x),(b',x'))
=
k_{\mathrm{shared}}(b,b')
+
\mathbf 1[b=b']\,k_b(x_{A(b)},x'_{A(b)}).
$$

The shared term is a valid kernel over branch identifiers; each
\(k_b\) is a valid kernel on its branch's active subspace.
The branch-specific term is block diagonal and hence positive
semidefinite. This construction permits cross-branch information
sharing through \(k_{\mathrm{shared}}\) while never comparing inactive
child coordinates.

**Not every heuristic that discards nonshared coordinates defines a
positive-semidefinite kernel.** A hierarchical distance formula must
be justified separately before inserting it into a standard radial
kernel.

### 2.2 Hierarchical multitask structure

For tasks \(t,t'\), a separable example is

$$
k((b,x,t),(b',x',t'))
=
k_{\mathrm{hier}}((b,x),(b',x'))\,k_{\mathrm{task}}(t,t').
$$

This is valid because products of positive-semidefinite kernels are
positive semidefinite. Separability is a modeling assumption:
different tasks may instead require task-specific hierarchical
kernels or nonseparable cross-task relationships.

## 3. Heterogeneous multitask covariates

Suppose task \(t\) observes only a subset \(S_t\) of an ambient
covariate vector. The observed covariate space is then
\(\mathcal X_t=\mathbb R^{|S_t|}\), and the statistical model must
specify how tasks with different observed coordinates share
information.

A common approach introduces a shared latent representation

$$
h_t:\mathcal X_t\to\mathcal Z,\qquad
f_t(x)=g_t(h_t(x)).
$$

One possible cross-task covariance is

$$
k((t,x),(t',x'))
=
k_{\mathcal Z}(h_t(x),h_{t'}(x'))
\,k_{\mathrm{task}}(t,t').
$$

For fixed maps \(h_t\), this is a valid kernel when both factors are
valid kernels. The maps and covariance assumptions determine which
information transfers. Merely padding missing features with zeros
does not preserve the meaning of unobserved covariates.

Negative transfer can occur if the shared representation or
cross-task covariance is misspecified.

## 4. Contextual additive models

Partition an input into context blocks \(x=(x_1,\ldots,x_C)\).
An additive latent function assumes

$$
f(x)=\sum_{c=1}^{C}f_c(x_c).
$$

If the components are independent GPs, their sum is a GP with

$$
k(x,x')=\sum_{c=1}^{C}k_c(x_c,x_c').
$$

This covariance does not capture interactions between contexts.
When interactions are important, additive assumptions can
underfit even if each component is modeled accurately.

If only the aggregate output \(f(x)\) is observed, individual
component functions may not be identifiable without additional
constraints or priors. If context-specific outputs are observed,
a multi-output model can exploit that richer observation structure.

## 5. Latent context embeddings

Let \(e_c\in\mathbb R^r\) be an embedding for context \(c\).
A shared GP construction may use

$$
k_{\mathrm{context}}(c,c')
=
\kappa(e_c,e_{c'}).
$$

The embeddings induce a positive-semidefinite context kernel
when \(\kappa\) is valid. They can encode graded context
similarities rather than forcing all contexts to be independent
or identical.

Jointly learned embeddings may be weakly identified: rotations,
translations (for stationary embedding kernels), or other
transformations can leave covariance predictions unchanged.
Regularization and held-out validation are important when the
number of contexts or observations is small.

## 6. Comparison and model selection

- **Hierarchy:** Is the validity or meaning of a coordinate
  conditional on a parent choice?
- **Heterogeneous tasks:** Do different tasks observe different
  covariate subsets, with a scientifically justified transfer
  relationship?
- **Contextual decomposition:** Can the response reasonably be
  expressed as a sum or structured interaction of context
  contributions?
- **Context embedding:** Is similarity among contexts itself
  a quantity to estimate?

The choice is driven by the data-generating assumptions, not by
the availability of a named model. Structured models may improve
sample efficiency when those assumptions hold and degrade
performance when they do not.

Hierarchical *candidate feasibility* is distinct from hierarchical
*covariance*: a surrogate can respect inactive coordinates without
guaranteeing that an optimizer generates only valid configurations.

## 7. References and further reading

- Rasmussen, C. E., & Williams, C. K. I. (2006).
  *Gaussian Processes for Machine Learning*. MIT Press.
- Duvenaud, D., Nickisch, H., & Rasmussen, C. E. (2011).
  Additive Gaussian Processes. *NeurIPS*.
- Bonilla, E. V., Chai, K. M. A., & Williams, C. K. I. (2008).
  Multi-task Gaussian Process Prediction. *NeurIPS*.
- Hutter, F., Hoos, H. H., & Leyton-Brown, K. (2011).
  Sequential Model-Based Optimization for General Algorithm
  Configuration. *LION*.

## Related documents

- [Mixed Variables](05_mixed_variables.md)
- [Multi-task / Multi-output](07_multitask_multioutput.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [Model Selection](13_model_selection.md)
- [Model documentation](../models/README.md)

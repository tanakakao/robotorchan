# Mixed one-shot acquisition optimization

## Scope

robotorchan supports runtime evaluation and correctness-first candidate optimization of
mixed one-shot acquisition functions when the underlying model supports the required
fantasy operation. This includes mixed qKG and mixed multi-fidelity qMFKG with categorical
values that may differ between the observed candidate and fantasy decision points.

## Why `optimize_acqf_mixed` is not sufficient for qKG

BoTorch one-shot knowledge-gradient acquisitions optimize an augmented batch. The first
rows represent the candidates that will actually be evaluated, while additional rows
represent fantasy decision points used internally by the acquisition.

`optimize_acqf_mixed` applies each entry in `fixed_features_list` to the complete
augmented batch. For a categorical feature, this therefore fixes the observed candidate
and every fantasy decision point to the same category.

That restriction is not generally valid. A fantasy posterior can have its optimum in a
different category from the category of the observed candidate.

Consequently, robotorchan must not present ordinary `optimize_acqf_mixed` as a correct
general optimizer for mixed qKG or mixed qMFKG.

## Current support

The current runtime contract is:

- mixed qMFKG model construction: supported;
- target-fidelity incumbent optimization: supported;
- cost-aware utility and target-fidelity projection: supported;
- one-shot acquisition evaluation: supported;
- different categories across fantasy decision points: supported;
- mixed one-shot candidate optimization for `q=1`: supported by
  `optimize_mixed_one_shot_acqf` using exact row-wise categorical enumeration;
- larger `q` and categorical assignment spaces above the configured exact-enumeration limit:
  intentionally unsupported.

Runtime coverage includes both explicit cross-category augmented batches and candidate
generation with real mixed qKG and mixed qMFKG acquisitions.

## Optimizer contract

The mixed one-shot optimizer preserves the distinction between actual candidates
and fantasy decision points. In particular, it must allow categorical assignments to vary
by augmented-batch row rather than applying one fixed assignment to every row.

The implementation also preserves BoTorch one-shot semantics for extracting the
actual candidates from the optimized augmented batch. Runtime tests must include a case
where the selected candidate category and at least one fantasy optimum category differ.

The current exact implementation is deliberately scoped to `q=1` and bounded finite
categorical assignment spaces. It fails explicitly outside that contract rather than
falling back to ordinary `optimize_acqf_mixed`.


## Optimizer design

The mixed one-shot optimizer will treat categorical assignments as part of the augmented
batch state rather than as global fixed features.

For an acquisition with actual batch size `q` and augmented size `q_aug`, define a
categorical assignment matrix with one assignment per augmented row. The optimizer must:

1. enumerate feasible categorical assignments for the actual candidate rows;
2. allow fantasy rows to use assignments independently from the actual rows;
3. optimize continuous coordinates conditionally on each augmented categorical assignment;
4. compare acquisition values across the resulting conditional optima;
5. return only the actual candidates using the acquisition's one-shot extraction semantics.

The first implementation will target `q=1` and finite categorical alternatives. This keeps
the combinatorial search explicit and correct before introducing pruning or approximate
search. Multi-category and larger-`q` acceleration are future performance extensions, not
requirements for semantic correctness.

### Correctness invariants

The implementation must satisfy all of the following:

- a fantasy row may select a different category from the observed candidate;
- categorical coordinates remain fixed during conditional continuous optimization;
- fidelity coordinates remain optimizable unless explicitly fixed by the caller;
- the acquisition receives the complete augmented batch;
- returned candidates exclude fantasy decision rows;
- the reported acquisition value corresponds to the optimized augmented batch;
- ordinary non-one-shot mixed acquisitions continue to use BoTorch's native mixed optimizer.

The implementation must fail explicitly when the categorical assignment space is too large
for the exact enumeration path rather than silently falling back to a semantically
restricted optimizer.

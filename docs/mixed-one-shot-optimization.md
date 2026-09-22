# Mixed one-shot acquisition optimization

## Scope

robotorchan supports runtime evaluation of mixed one-shot acquisition functions when the
underlying model supports the required fantasy operation. This includes mixed
multi-fidelity qKG evaluation with categorical values that differ between the observed
candidate and fantasy decision points.

Candidate optimization is a separate capability.

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
- general mixed one-shot candidate optimization: not yet implemented.

The runtime test intentionally constructs the augmented batch explicitly so that fantasy
decision points exercise categories independently from the observed candidate.

## Requirements for a future optimizer

A future mixed one-shot optimizer must preserve the distinction between actual candidates
and fantasy decision points. In particular, it must allow categorical assignments to vary
by augmented-batch row rather than applying one fixed assignment to every row.

An implementation should also preserve BoTorch one-shot semantics for extracting the
actual candidates from the optimized augmented batch. Runtime tests must include a case
where the selected candidate category and at least one fantasy optimum category differ.

Until these requirements are satisfied, mixed qKG and qMFKG should be treated as
evaluation-compatible but not as having a general-purpose mixed candidate optimizer.

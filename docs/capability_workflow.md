# Capability-aware workflow

robotorchan は、モデル数を増やすだけでなく、問題要件と実装能力を明示的に対応付けるための capability-aware layer を提供します。

## 流れ

ProblemSpec → Model Registry → Selector → Model × Acquisition Compatibility → Recommendation → Capability Benchmark

ProblemSpec は問題の構造を宣言します。モデル名を直接指定する設定ではありません。

    from robotorchan.models import InputType
    from robotorchan.workflow import (
        ProblemPurpose,
        ProblemSpec,
        recommend_compatible_workflows,
    )

    spec = ProblemSpec(
        purpose=ProblemPurpose.BAYESIAN_OPTIMIZATION,
        input_type=InputType.MIXED,
        high_dimensional=True,
    )
    recommendations = recommend_compatible_workflows(spec)

## ProblemSpec の意味

- input_type: continuous / mixed
- task_type: single-task / multi-task
- output_type: single-output / multi-output
- objective_type: BO における single-objective / multi-objective
- multi_fidelity: fidelity 構造の有無
- structured_output: structured posterior を必要とするか
- high_dimensional: 高次元向け戦略を必要とするか
- robust: robustness strategy を必要とするか
- preference: pairwise preference observation を使うか
- q: 同時に評価する候補数。1以上
- constrained: BOで制約付き最適化を宣言するか

q と constrained は ProblemSpec の契約として保持します。Active Learning / Bayesian Optimization の acquisition metadata と Recommendation が q、制約、目的数を照合します。

output_type, objective_type, task_type は別概念です。Active Learning の multi-output を multi-objective BO として扱いません。BO で objective_type=multi を指定する場合は output_type=multi が必須です。単目的 BO では multi-output posterior を scalarizing objective で単一目的へ写像する構成も扱えます。

## Selector

select_compatible_models() は registry metadata に基づく構造的なフィルタです。モデル名の文字列解析によって capability を推測しません。evaluate_models() を使うと、不適合モデルについて理由も取得できます。

## Recommendation

recommend_compatible_workflows() は互換候補を返しますが、モデル性能のランキングや「最良モデル」の判定は行いません。

Active Learning では robotorchan の acquisition registry と model-acquisition compatibility を組み合わせます。multi-output 問題では single-output 限定 acquisition を除外します。

Bayesian Optimization では BoTorch 標準 acquisition を robotorchan 側で再実装しません。代表的な BoTorch-native acquisition を metadata-only entry として registry に保持し、objective_type、output_type、constrained、q、ensemble posterior、fantasy-model requirement を Recommendation で照合します。実際の acquisition 実装は BoTorch のクラスを使用します。q 系の Monte Carlo acquisition は posterior mean / variance だけでなく posterior sample を利用するため、registry でも posterior_samples requirement として明示します。モデル側にも supports_posterior_samples capability を持たせ、model-acquisition compatibility でこの契約を検証します。

## Benchmark

robotorchan.benchmarks.capability は予測精度 benchmark ではなく、capability engine の整合性を検証するための構造 benchmark です。registry 全体に対して ProblemSpec のフィルタと Recommendation が一貫していることを確認します。

モデル自体の予測性能や BO performance benchmark とは目的を分離しています。

## Source of truth

モデル capability の内部 source of truth は `MODEL_REGISTRY`、Acquisition metadata の内部
source of truth は `ACQUISITION_REGISTRY` です。利用者は内部dictへ直接依存せず、
`robotorchan.models.get_model_registry_entry()` と
`robotorchan.acquisition.get_acquisition_registry_entry()` からimmutable metadataを取得します。
`docs/model_coverage.json` はmodel registryから生成されます。

新しいモデルや acquisition を追加するときは、実装だけでなく対応する capability metadata と contract test も更新してください。


## Runtime validation

Capability metadata describes structural support. It is not a model-wide claim that every
compatible workflow has been executed end to end. Representative executable coverage is tracked by executable tests and benchmark harnesses rather
than a point-in-time development matrix.

When extending the library, keep these layers distinct:

- registry capability: the model or acquisition exposes the required structural behavior;
- compatibility: the model and acquisition contracts are not known to conflict;
- runtime validation: a representative executable workflow has been tested;
- benchmark evidence: comparative quality or performance has been measured.

Do not infer runtime validation or benchmark superiority from registry membership alone.

## Component selection

Use `evaluate_components(spec)` to inspect every factory-visible model,
acquisition, optimizer and search strategy. Results are independent component
checks, not compatible workflow tuples or performance rankings. Status values
are compatible, incompatible, conditional and unknown. Unknown means metadata
is missing; conditional means runtime or cross-component checks are required.
`ProblemSpec.observation_kind` distinguishes regression, binary, multiclass and
ordinal observations. `candidate_constrained` is separate from outcome
constraints (`constrained`). Model-only selection uses the regression registry by default and classification
registry IDs for binary, multiclass and ordinal problems. Both selectors share
structural checks. Recommendation accepts these registry IDs for static
model/acquisition checks; full four-component compatibility is a subsequent phase.


Factory names without reliable metadata remain unknown, including generic
wrappers whose capabilities depend on their supplied base model. This is not
an assertion that the implementation is unsupported. Acquisition limitations
are included in conditional selection reasons.

Specialized constructors are also classified when their contracts are known:
BOCS and COMBO require binary design coordinates; RGPE requires scalar Gaussian
source/target models; causal surrogates require intervention data and a known
DAG. SCMs are not native BoTorch models. Calibrated classifiers and generic
classification ensembles require checks of the supplied base estimator or
members. These entries are conditional, not automatically recommended.

Classifier feasibility adapters are evaluated separately from single-model
acquisition metadata. ClassificationProbabilityAcquisition is scalar and q=1;
FeasibilityWeightedAcquisition requires both an objective acquisition and a
classifier feasibility adapter, inherits the wrapped acquisition's q and
objective limits, and rejects pending candidates and joint fantasization.
The product reduction assumes independent feasibility probabilities.

Additional native acquisition contracts cover analytic constrained EI, posterior
standard deviation, and negative integrated posterior variance when exposed by
the installed BoTorch version. The latter requires fantasy conditioning and
integration points. HeterogeneousModel remains conditional on its members.
To inspect remaining unknown factory names and counts, run:

```bash
pytest -s tests/workflow/test_component_selection.py -k report_remaining_unknown
```

The final inventory gate requires every factory name to have an explicit
static decision or a constructor-dependent condition. Native projection,
scalarization, feasibility, probability of improvement, confidence bounds,
pairwise variance and lower-bound entropy acquisitions are included. This
is not a guarantee of runtime compatibility or a count of unique classes.
Unrecognized external names still return UNKNOWN.


### Integrated workflow compatibility

`check_compatibility(spec, model=..., acquisition=..., optimizer=...,
search_strategy=...)` combines the component checks with registered posterior
and one-shot optimizer contracts. Reasons describe rejection; conditions
describe unresolved runtime requirements. It does not fit or construct models.

`recommend(spec, model=..., acquisition=..., optimizer="botorch")` returns
unranked candidates with the compatibility result. Explicit choices are not
silently replaced. The default optimizer is BoTorch, not a performance ranking.
An empty result means no candidate passed these checks. Use the compatibility
function to diagnose an explicitly rejected combination.


Explicit model class names and special acquisition names are preserved even
when only one component is specified. Automatic enumeration uses registered
models and acquisitions. When a strategy is specified without an optimizer,
the strategy owns optimization; no dispatcher default is injected. Explicit
optimizer choices are accepted only for matching BoTorch strategy paths or
the continuous TuRBO dispatcher interface.


Use `optimizer="auto"` or `search_strategy="auto"` to enumerate the respective
factory inventory, rather than silently choosing a backend. Results remain
unranked and conditional; explicit names are preserved. Both auto options
enumerate their combinations, so specify a model and acquisition to limit
the result size. Dedicated causal surrogates require their own workflow
or native model extraction; vector NSGA-II is not a scalar acquisition backend.


Explicit `model="auto"` and `acquisition="auto"` include all factory names,
including special conditional components. Omitted names retain the registered
default inventory. Class names and registry IDs may refer to the same class.
`WorkflowRecommendation.required_arguments()` inspects the original
constructor or optimizer signature on demand, without constructing a model.
It lists required named parameters, not their values or conditional kwargs;
None means signature inspection is unavailable. Runtime conditions still apply.


Integration tests exercise recommendations through the original factories,
explicit fit calls, acquisition construction and candidate generation for
single-objective, multi-objective, outcome-constrained, mixed candidate-constrained,
binary, multiclass and ordinal problems. These are smoke tests, not performance
benchmarks or exhaustive validation of every conditional combination.


### Study integration

The existing scalar-objective Study accepts exact model and acquisition factory
names as well as the original callable constructors. Names are resolved at
Study construction, before initial trials; training remains in the existing
candidate generation loop. This does not extend Study to vector observations
or classification active learning.

```python
from robotorchan.study import create_study

study = create_study(
    search_space={"x": (0.0, 1.0)},
    model="SingleTaskGP",
    acquisition="qLogExpectedImprovement",
    optimizer="sobol",
)
for _ in range(5):
    trial = study.ask()
    study.tell(trial, -((trial.params["x"] - 0.6) ** 2))
```


Study also accepts canonical SearchStrategy names through its existing optimizer
argument. For example, `optimizer="SobolSearchStrategy"` with
`optimizer_kwargs={"num_samples": 32}` constructs the original strategy once
using Study bounds. An existing strategy instance remains valid. Scalar
dispatcher names use the optimizer factory; specialist APIs with different
signatures or vector results are not accepted as Study optimizer names.
The existing restrictions on strategy instances (mixed domains, constraints,
and context-fixed proposals) still apply.

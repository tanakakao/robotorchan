# Capability-aware workflow

robotorchan は、モデル数を増やすだけでなく、問題要件と実装能力を明示的に対応付けるための capability-aware layer を提供します。

## 流れ

ProblemSpec → Component selection → Workflow compatibility → Recommendation → Explicit factory construction / Study

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
- constrained: BOで未知の出力制約を宣言するか
- candidate_constrained: 既知の候補入力制約を宣言するか
- observation_kind: regression / binary / multiclass / ordinal

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
model/acquisition checks; full four-component compatibility is available through check_compatibility().


Unrecognized names without reliable metadata remain unknown. Generic
wrappers with known constructor-dependent contracts are conditional. This is not
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


A selected `WorkflowRecommendation` can be supplied to `create_study` using
`recommendation=selection` instead of explicit model, acquisition and optimizer
names. Study rechecks the scalar regression BO contract for its domain and
candidate constraints at construction. This is a q=1 static check, not a
guarantee for later batch sizes or constructor arguments. Supply required
method-specific kwargs explicitly. Recommendations specifying both an optimizer
and a search strategy are rejected rather than silently dropping either.


The existing single-objective benchmark callback also accepts factory names:

```python
from robotorchan.benchmarks.botorch_strategy import make_botorch_gp_strategy

strategy = make_botorch_gp_strategy(
    model_factory="SingleTaskGP",
    acquisition="qLogExpectedImprovement",
    optimizer="sobol",
    raw_samples=64,
)
```

Named models use their existing fit method. The original callable model factory
retains native exact GP fitting, and the legacy acquisition names and default
native optimizer remain supported. This callback remains restricted to
unconstrained continuous scalar regression BO; other benchmark paths are unchanged.


Study is in-memory; benchmark checkpoint/storage is a separate API.
To resume from historical observations, construct a Study with the same
explicit component configuration (or selected recommendation) and pass
`initial_trials=previous_study.trials`. Trial IDs, states and observations
are retained, including running and waiting trials. The next model fit uses
completed observations only. This is history restoration, not a persistent
storage backend or exact RNG/strategy checkpoint. The caller must retain the
configuration and history externally. Historical joint-batch feasibility
cannot be reconstructed from individual trial records and is not rechecked;
parameter bounds and discrete domains are validated.


### Study and benchmark integration boundary

The Study integration tests cover selected recommendations with either a scalar
dispatcher or a named Sobol search strategy, minimization, model-based q-batch
proposals, trial history restoration and continued optimization. The benchmark
integration test exercises the named model/acquisition/optimizer path with q=2.
These are representative runtime contracts, not exhaustive cross-product tests.
Persistent Study storage, exact checkpoint replay and specialized multi-output,
classification or causal Study loops are outside this integration boundary.


## 統一Factoryによる候補提案

Factoryは既存クラスを生成する入口であり、学習や候補提案を暗黙に実行しません。

```python
import torch
from robotorchan.models import create_model, list_models
from robotorchan.acquisition import create_acquisition, list_acquisitions
from robotorchan.optim import create_optimizer, list_optimizers

X = torch.linspace(0.1, 0.9, 6, dtype=torch.double).unsqueeze(-1)
Y = -((X - 0.6) ** 2)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
model = create_model("SingleTaskGP", X, Y)
model.fit()
acq = create_acquisition("qLogExpectedImprovement", model=model, best_f=Y.max())
optimize = create_optimizer("sobol", raw_samples=64, seed=17)
candidates, values = optimize(acq, bounds, q=2)
```

文字列には公開Factoryの一覧にある名前を使います。従来のクラスを直接
importして構築する方法も引き続き利用できます。Optimizer Factoryは
呼び出し可能な最適化関数を返します。Search Strategy Factoryは
状態を保持し得る既存のStrategyインスタンスを返す点が異なります。

## 問題設定から構成を選ぶ

```python
from robotorchan.workflow import ProblemSpec, evaluate_components, recommend
from robotorchan.study import create_study

spec = ProblemSpec(purpose="bayesian_optimization", q=2)
inventory = evaluate_components(spec)
(selection,) = recommend(
    spec,
    model="SingleTaskGP",
    acquisition="qLogExpectedImprovement",
    optimizer="sobol",
)
print(selection.compatibility)
print(selection.required_arguments())
study = create_study(
    search_space={"x": (0.0, 1.0)},
    recommendation=selection,
    optimizer_kwargs={"raw_samples": 64, "seed": 17},
    seed=17,
)
study.optimize(lambda trial: -((trial.params["x"] - 0.6) ** 2), n_trials=3)
for trial in study.ask_batch(2):
    study.tell(trial, -((trial.params["x"] - 0.6) ** 2))
```

判定は compatible / conditional / incompatible / unknown の4種類です。
conditional は追加の引数やposterior等の条件が残ることを示します。
recommend は性能順位ではなく静的な適合候補を返します。
Studyは単目的のスカラー観測を対象とし、すべてのFactory構成を
そのままStudyへ渡せるわけではありません。

実装構造については[Guide](guide/13_model_selection.md)を参照してください。


## 公開APIリファレンス

| 公開モジュール | API | 戻り値・役割 |
|---|---|---|
| robotorchan.models | `list_models()`, `get_model_class(name)`, `create_model(name_or_callable, *args, **kwargs)` | 公開名一覧、元のクラス、未学習のモデル |
| robotorchan.acquisition | `list_acquisitions()`, `get_acquisition_class(name)`, `create_acquisition(name_or_callable, *args, **kwargs)` | 公開名一覧、元のクラス、獲得関数 |
| robotorchan.optim | `list_optimizers()`, `create_optimizer(name_or_callable, **kwargs)` | 公開名一覧、設定済みCallable |
| robotorchan.optim | `list_search_strategies()`, `create_search_strategy(name_or_callable, *args, **kwargs)` | 公開名一覧、元のStrategyインスタンス |
| robotorchan.workflow | `ProblemSpec`, `evaluate_components(spec)` | 問題要件、個別コンポーネントの判定 |
| robotorchan.workflow | `check_compatibility(spec, model=..., acquisition=..., optimizer=..., search_strategy=...)` | WorkflowCompatibility |
| robotorchan.workflow | `recommend(spec, ...)` | WorkflowRecommendationの候補集合 |
| robotorchan.study | `create_study(..., recommendation=..., initial_trials=...)` | スカラー目的のStudy |

Factoryの識別子は位置引数で渡します。モデル固有の引数は元のコンストラクタへ渡されます。
公開名一覧には登録IDとクラス名の両方が含まれる場合があります。
`required_arguments()` は必要な引数名を調べるもので、値を自動決定しません。
専用の因果・安全・多目的ベクトル探索などは固有APIの契約を維持します。

## 実行可能なNotebook

- [62: 個別Factory](../examples/notebooks/62_unified_factories.ipynb)
- [63: Capability・Recommendation](../examples/notebooks/63_capability_recommendation.ipynb)
- [64: Study・Ask-and-Tell・Benchmark](../examples/notebooks/64_unified_study_benchmark.ipynb)

Notebook 64のDataFrame出力には `dataframe` extraが必要です。
`pip install -e ".[examples,dataframe]"` で実行環境を準備できます。

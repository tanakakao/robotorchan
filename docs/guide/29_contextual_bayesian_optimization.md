# 29. Contextual Bayesian Optimization：robotorchan実装ガイド

[理論](../theory/29_contextual_bayesian_optimization.md) / [Benchmark](../../benchmarks/CONTEXTUAL_POLICY.md)

## 1. APIの対応表

| 目的 | API | 注意点 |
| --- | --- | --- |
| Contextを固定した獲得関数最適化 | `robotorchan.optim.contextual.ContextualOptimizer` | `decision_indices`と`context_indices`を分離 |
| Context付きの試行管理 | `robotorchan.study.create_study(context_names=(...))` | Context列もGP学習入力に含める |
| 1 Contextで候補提案 | `Study.ask(context={"ambient": 0.3})` | Contextは範囲内で指定 |
| 同一Contextのq候補 | `Study.ask_batch(q, context=...)` | 同一Contextのjoint q最適化 |
| 複数の異なるContext | `Study.ask_contexts([...])` | 各Contextを独立にq=1最適化 |
| 実測値を登録 | `Study.tell(trial, value)` | 目的関数の実測値を登録 |
| 有限Context分布で方策を評価 | `expected_policy_value(outcomes, weights)` | 結果はユーザーが用意 |
| Monte Carlo方策評価 | `monte_carlo_policy_value(..., n_samples=..., generator=...)` | 同一サンプルを方策間で共有 |

## 2. 最小のAsk-and-Tell例

```python
from robotorchan.study import create_study

study = create_study(
    search_space={"temperature": (0.0, 1.0), "ambient": (0.0, 1.0)},
    context_names=("ambient",),
    n_initial_trials=3,
    seed=7,
)

for ambient in (0.2, 0.8, 0.4, 0.6, 0.3):
    trial = study.ask(context={"ambient": ambient})
    temperature = trial.params["temperature"]
    measured_quality = 1.0 - (temperature - ambient) ** 2
    study.tell(trial, measured_quality)
    print(trial.number, trial.params, measured_quality)
```

このコードでは、`ambient`を操作せず、観測した値を固定して`temperature`だけを提案します。3件の初期候補の後は、既存のGP・獲得関数・optimizerを使って候補を提案します。実務では`measured_quality`を実験装置から得た実測値に置き換えます。

## 3. 有限Context分布での評価

```python
import torch
from robotorchan.optim.contextual_policy import expected_policy_value

outcomes = torch.tensor([[0.8, 0.7], [0.9, 0.9]], dtype=torch.double)
weights = torch.tensor([1.0, 3.0], dtype=torch.double)
print(expected_policy_value(outcomes, weights))
```

行は方策、列はContextです。重みは内部で正規化されます。GP事後分布や候補最適化はこの関数では行いません。

## 4. Mixed・制約・q-batchの境界

`Study`は既存の`MixedVariableSpace`と`CandidateConstraints`を再利用します。Context値は`fixed_features`へ変換され、候補最適化時に固定されます。異なるContextの`ask_contexts`は候補間制約をサポートしません。Mixedで使えるoptimizerやfixed-feature列挙の条件は既存のoptimizer側の契約に従います。

## 5. 性能比較をどう読むか

[ベンチマーク](../../benchmarks/CONTEXTUAL_POLICY.md)では、既知の理論最適方策との比較と、実際のStudyによる同一評価予算の探索的比較を区別しています。学習中の平均観測値を未知Contextでの汎化性能と解釈しないでください。

# End-to-End実装ウォークスルー：観測から次の評価へ

このページは、[理論のBO](../../theory/01_bayesian_optimization.md)と[実装ガイドのBO](../01_bayesian_optimization.md)で説明した処理を、**実際に参照できるコード**に接続する。単なるAPI一覧ではなく、データがどの時点で何に変換されるかを追う。独立した49番目の理論章は作らず、固定48章に対する横断補助資料として配置する。

## 1. 一周の責務を確認する

```text
実験済み (train_X, train_Y)
          ↓  データの尺度・入出力型を検査
      surrogate model
          ↓  likelihood/MLLに対応したfit
      posterior(X)
          ↓  objectiveを通したutility
      acquisition(X)
          ↓  bounds、q、candidate constraintを考慮
      optimizer
          ↓  提案候補 [q, d]
      expensive evaluation (new_Y)
          ↓  観測済みデータへappend
    次iteration: 再fit・再提案
```

予測値、獲得関数値、真の評価値は異なる。特に`acquisition_value`を観測`new_Y`として学習データへ入れてはいけない。

## 2. 単目的・連続BO：最小のclosed loop

以下は`SingleTaskGP`の公開`fit()`、BoTorchのLogEI、robotorchanのSobol optimizerを接続した**読める最小例**。高価な評価関数の代わりに決定論的な合成関数を使用する。BoTorchの`LogExpectedImprovement`はGaussian posteriorのsingle-output、`q=1`を前提とする。

```python
import torch
from botorch.acquisition.analytic import LogExpectedImprovement

from robotorchan.models import SingleTaskGP
from robotorchan.optim import optimize_acqf


def expensive_objective(X: torch.Tensor) -> torch.Tensor:
    return -((X - 0.72) ** 2).sum(dim=-1, keepdim=True)


torch.manual_seed(42)
bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
train_X = torch.tensor([[0.1], [0.4], [0.9]], dtype=torch.double)
train_Y = expensive_objective(train_X)

for iteration in range(3):
    model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
    model.fit(optimizer_kwargs={"options": {"maxiter": 25}})
    posterior = model.posterior(train_X)
    assert posterior.mean.shape == train_Y.shape

    acquisition = LogExpectedImprovement(model=model, best_f=train_Y.max())
    candidate, score = optimize_acqf(
        acquisition,
        bounds,
        q=1,
        optimizer="sobol",
        raw_samples=64,
        seed=42 + iteration,
    )
    assert candidate.shape == (1, 1)
    assert torch.isfinite(score).all()
    new_Y = expensive_objective(candidate)
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_Y], dim=0)

print(train_X.shape, train_Y.shape)
```

このコード例は既存の`SingleTaskGP`とoptimizer APIに沿って記したものだが、**今回の作業で実際に実行した証拠ではない**。実行速度、学習収束、同一candidateの重複、正規化やノイズの扱いなどは目的に応じて調整する。学習・候補選択時に同じ`double` dtypeを使用し、最大化の`best_f`は**最大化する目的の尺度**で指定する。

### ソースで追跡する

`src/robotorchan/models/standard/single_task.py`の`SingleTaskGP`はBoTorchからposteriorを継承し、raw観測保持と`fit()`の便宜機能を追加する。`LogExpectedImprovement`はBoTorch-nativeであり、robotorchan独自のEI再実装ではない。`robotorchan.optim.optimize_acqf`はoptimizer文字列に応じて候補探索をdispatchする。実験評価関数の実行、観測のappend、iteration制御は**利用側の責務**である。

## 3. 多出力・batch：既存の実行スクリプトへつなぐ

[unified_api_workflow.py](../../../examples/unified_api_workflow.py)は、2出力`SingleTaskGP`から`posterior.mean[3,2]`を取り、BoTorch `LinearMCObjective`の重み`[0.7,0.3]`でスカラーutilityへ変換する。`qSimpleRegret`と`SobolQMCNormalSampler`によりjoint `q=2`の獲得関数を作り、`optimize_acqf(..., optimizer="sobol")`で`[2,1]`候補を返す。このサンプルは**候補提案まで**で、実験評価と更新を行う完全closed-loopではない。

```bash
python examples/unified_api_workflow.py
```

多出力であることはPareto multi-objectiveを意味しない。どの出力をスカラー化するか、`q`のjoint posterior依存をどう評価するかを明示する。詳細は[統合APIガイド](../../optimization/unified-api-workflow.md)。

## 4. Mixed Variable BO

連続設計列とカテゴリ列を分離して`MixedSingleTaskGP(train_X, train_Y, cat_dims=[...])`を構築する。カテゴリは順序を持つ連続値として扱わない。optimizer側はカテゴリ値の組合せを`fixed_features_list`で渡すBoTorch `optimize_acqf_mixed`などを選択し、**元のカテゴリコード**を保って真の評価関数へ渡す。

学習側のnative categorical covarianceと候補生成側のcategory assignmentは別の契約である。モデルに`cat_dims`を渡すだけでoptimizerがカテゴリ候補を自動列挙する、とは仮定しない。使用例は[Mixed GP](../05_mixed_variables.md)と[Mixed最適化](../optimization/04_mixed_discrete.md)、[Mixed Notebook](../../../examples/notebooks/02_mixed_single_task_gp.ipynb)を参照する。

## 5. 多目的・制約付きBO

多目的では1点の出力`Y[..., m]`に複数の目的が含まれる。Pareto dominance・reference point・partitioningを正しい**objective尺度**で定義し、BoTorchのEHVI/NEHVI系の対応acquisitionへ渡す。多出力GPの存在だけでEHVIが自動構築されることはない。参照：[多目的獲得関数](../acquisition/07_multiobjective.md)。

未知のoutcome constraintはsurrogateとacquisition側で可行性を評価する。一方、`x`だけで判定できるcandidate constraintはoptimizer側の`CandidateConstraints`で扱う。制約の符号は層ごとに異なり得るので、candidate inequality `g(x) >= 0`とoutcome constraint `c(y) <= 0`を混同しない。参照：[制約付き獲得関数](../acquisition/08_constraints.md)、[optimizer制約](../optimization/05_constraints.md)。

## 6. 回帰＋分類の異種モデル

回帰目標と二値分類可行性を同じ`posterior()`へ連結して済ませない。分類モデルの`posterior()`は潜在関数であり、`predict_proba()`がclass probabilityである。`HeterogeneousModel`は回帰と分類の出力索引を統合管理するが、Gaussian latentとclass probabilitiesを混ぜた共通posteriorは作らない。

`ProblemSemantics`で目的の向きと可行性を定め、`src/robotorchan/acquisition/composition.py`のsemantic bridgeから適切な獲得関数を構築する。分類可行確率をq点で単純に乗じる構成には、**条件付き独立性近似**が入る場合がある。実装とテストの導線は[Unified API](../../optimization/unified-api-workflow.md)および[回帰＋分類の結合テスト](../../../tests/integration/test_phase9_mixed_heterogeneous.py)で確認できる。閉ループNotebookは[End-to-End workflows](../../../examples/notebooks/36_end_to_end_workflows.ipynb)。

## 7. 再現・検証チェックポイント

| チェック | 検査する値 | 検査範囲 |
| --- | --- | --- |
| 学習 | modelのfit契約、finite loss | fit可能モデルと学習方式の一致 |
| posterior | `mean`と`variance`のshape・finite | predictive semantics; calibrationとは別 |
| acquisition | input shape、finite score、目的の向き | 数値的安定性; 最適性とは別 |
| optimizer | `candidate.shape == (q,d)`、bounds | candidate feasibility; global optimumとは別 |
| 新観測 | expensive objectiveからの`new_Y` | scoreをYとしてappendしない |
| 更新 | 元`X/Y`へのappend・再fit | 学習セットとmodel versionを揃える |
| heterogeneous | 回帰潜在値と分類確率を別に取得 | independence approximationの範囲 |

既存の[Model Mapping](../../../.github/project-records/theory-guide/mapping/models.md)、[Acquisition Mapping](../../../.github/project-records/theory-guide/mapping/acquisition.md)、[Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md)、[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)は移行元・検証の証拠として保全する。本ページは原文各区間の完全移行や全例の実行成功を主張しない。数学的referenceは[Theory索引](../../theory/README.md)から辿れる。

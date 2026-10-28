# 多目的獲得関数

BoTorch 標準 API を直接使い、モデルとの接続をテストします。ローカルな alias は設けません。

## Hypervolume improvement

ノイズのない観測には `qLogExpectedHypervolumeImprovement`（qLogEHVI）、ノイズのある観測には `qLogNoisyExpectedHypervolumeImprovement`（qLogNEHVI）を推奨します。

```python
from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
)

acqf = qLogExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    partitioning=partitioning,
)
```


計上対象の目的値が reference point を支配するよう設定します。問題固有の意味を持つため、自動推定しません。

## qLogNParEGO

hypervolume の扱いが難しい場合や分解型探索を選ぶ場合、BoTorch `qLogNParEGO` によるスカラー化を利用できます。

```python
from botorch.acquisition.multi_objective.parego import qLogNParEGO

acqf = qLogNParEGO(
    model=model,
    X_baseline=train_X,
    scalarization_weights=weights,
)
```


重みは一つのスカラー部分問題を定めます。通常は BO の反復ごとに変化させ、Pareto front を探索します。

## 多目的 Knowledge Gradient

BoTorch の hypervolume KG は非近視眼的な選択肢ですが、fantasy 木の最適化は EHVI・NEHVI より重くなります。追加費用に見合う lookahead の価値を確認してください。独自 wrapper は設けません。

## 制約と custom objective

registry の `supports_constraints` は BoTorch の合成経路への対応を表します。制約 callable、実行可能性の符号、objective、reference point を自動構築する意味ではありません。

qLogEHVI・qLogNEHVI・qLogNParEGO を多目的経路として登録しています。qLogNParEGO は実行テストと capability registry の両方で表現され、文書・実行範囲・推奨 metadata を一致させています。

## 候補制約と出力制約

- 獲得関数の `constraints=[...]` は posterior sample の出力空間で `c(Y) <= 0` を実行可能とします。
- `CandidateConstraints.nonlinear_inequality_constraints` は候補最適化時の入力を評価し、`g(X) >= 0` を実行可能とします。

両方を同時に使えます。qLogEHVI・qLogNEHVI の出力実行可能性と、`optimize_acqf_botorch` による入力の制限は独立です。候補制約は目的や reference point を変更せず、出力制約は入力の実行可能性検査を代替しません。

点内制約は各候補 X に適用し、点間制約は単一目的と同じ joint-q の意味を保ちます。

## Hypervolume Knowledge Gradient

HVKG は BoTorch に存在しますが、このリポジトリには実行可能な統合検証がありません。現時点では将来の統合対象であり、実行検証済みとは扱いません。registry で対応を宣言するには、モデル互換性、fantasy の意味、候補最適化のテストが必要です。

## 検証範囲

標準の検証済み範囲は qLogEHVI・qLogNEHVI・qLogNParEGO です。制約と custom objective は明示的な BoTorch 入力です。HVKG と多忠実度 hypervolume lookahead は含みません。[多目的理論](../theory/acquisition/07_multiobjective.md)も参照してください。

# Model → Acquisition → Optimizer：公開APIと所有権

この補助資料は[48章の読書入口](../../../docs/guide/README.md)から、
モデル・獲得関数・候補最適化を接続する際の実装契約を確認するためのものです。
数理的な導出は対応する理論章を正本とし、ここでは実装の境界を説明します。

## 通常の単一目的・連続入力

| レイヤー | 入口 | ソースの所有権 | 出力と意味 |
| --- | --- | --- | --- |
| Model | `robotorchan.models.SingleTaskGP` | `src/robotorchan/models/standard/single_task.py`：BoTorch `SingleTaskGP`を継承 | 元空間入力`X`に対するposterior |
| Acquisition | `botorch.acquisition.analytic.UpperConfidenceBound` | BoTorch-native | 候補点に対するutility。観測済みYではない |
| Optimizer | `robotorchan.optim.optimize_acqf` | `src/robotorchan/optim/dispatch.py` | `(candidate, acquisition_value)` |
| BoTorch backend | `optimizer="botorch"` | `src/robotorchan/optim/backends/botorch.py`からBoTorchへ委譲 | `candidate[q,d]`を返す |

モデルの学習と候補探索は別の最適化問題です。獲得関数の出力値を
ブラックボックスの新規観測`train_Y`として追加しないでください。
実際の実験・シミュレーションで候補を評価し、観測を得てからモデルを更新します。

## 例外となる経路

| 問題 | 通常経路と異なる理由 | 読む章 |
| --- | --- | --- |
| 既知の候補制約 | `CandidateConstraints`はoptimizer側の可行域を定義 | [Constraints](../../../docs/guide/optimization/05_constraints.md) |
| 未知の出力制約 | 目的・可行性はacquisition側に反映 | [Constrained acquisition](../../../docs/guide/acquisition/08_constraints.md) |
| Mixed / categorical | native mixedの固定カテゴリ列挙は通常の連続dispatcherと異なる | [Mixed](../../../docs/guide/optimization/04_mixed_discrete.md) |
| qKG / one-shot | 拡張batchにfantasy decision variableを含む。実候補の抽出が必要 | [Batch / one-shot](../../../docs/guide/optimization/08_batch_one_shot.md) |
| 高次元embedding | 内部探索座標Zと実験座標Xが異なる | [High-dimensional](../../../docs/guide/optimization/09_high_dimensional.md) |
| TuRBO | 観測によるstate更新を持つsearch strategyで、named backendではない | [Trust region](../../../docs/guide/optimization/10_trust_region.md) |

## APIの照合手順

1. 公開入口のシグネチャを確認し、`model`、`bounds`、`q`等の引数を特定する。
2. 関数の呼出先を追跡し、robotorchan独自実装とBoTorchへの委譲を分ける。
3. `[n,d]`、`[...,q,d]`、`[q,d]`等のshapeと出力の意味を確認する。
4. 制約・mixed・fantasy・入力変換など、通常経路から外れる条件を確認する。
5. 掲載例テストと独立数値参照を区別して読む。

検証状況は[モデル](../../../.github/project-records/theory-guide/model-chapter-verification-matrix.md)、
[獲得関数](../../../.github/project-records/theory-guide/acquisition-chapter-verification-matrix.md)、
[最適化](../../../.github/project-records/theory-guide/optimization-chapter-verification-matrix.md)
の各マトリクスに記録しています。これは代表例のAPI照合であり、
全組合せの互換性や数値精度を証明するものではありません。

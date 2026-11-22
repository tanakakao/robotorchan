# 異種回帰・分類モデルによる制約付き候補提案

[理論と前提](../theory/32_heterogeneous_classification_constraints.md)

## 部品の対応

| 役割 | 公開API |
| --- | --- |
| 回帰・分類モデルをまとめる | `HeterogeneousModel` |
| 最大化する回帰出力 | `RegressionObjective` |
| 許容クラス集合 | `ClassificationConstraint(feasible_classes=(...))` |
| 目的と制約の宣言 | `ProblemSemantics` |
| 確率重み付きqEI | `make_classification_constrained_qei_acquisition` |
| 候補提案 | `robotorchan.optim.optimize_acqf` |

## 処理の流れ

1. `HeterogeneousModel` が回帰出力と分類出力の所有モデルを管理します。
2. `ProblemSemantics` が出力名と許容クラス集合を解決します。
3. 回帰側はBoTorchの `qExpectedImprovement` を構築します。
4. 分類側は `predict_proba` の許容クラス確率を合計します。
5. `FeasibilityWeightedAcquisition` が回帰qEIに分類確率を掛けます。
6. `optimize_acqf` がこのスカラー獲得関数を最適化します。

## 注意点

- `probability_threshold` を指定した制約は、この獲得関数ビルダーでは扱えません。確率重み付けと閾値制約は別物です。
- q>1 の積は独立性に基づく近似で、分類器の候補間相関は反映しません。
- Mixed変数では `MixedVariableSpace` と対応するOptimizerを使用します。実装テストで確認したのはSobolによる連続＋整数変数の候補提案です。すべてのモデル・Optimizer・カテゴリ変数の組み合わせを保証しません。
- `X_pending` と異種モデルのjoint fantasizationには対応しません。
- 分類器のクラスラベルと確率列の対応を必ず確認してください。

## 関連資料

- [分類モデルの理論](../theory/24_classification.md)
- [候補制約のガイド](optimization/05_constraints.md)
- [Mixed候補最適化](optimization/04_mixed_discrete.md)

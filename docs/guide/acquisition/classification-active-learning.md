# 分類 Active Learning の実装条件

[理論](../../theory/acquisition/classification-active-learning.md)で潜在・確率・label の不確実性を区別した後、実装の入力を確認します。

## 基準と予測 API

- `PredictiveEntropy`：posterior predictive の class entropy。
- `MarginUncertainty`：上位 2 class の曖昧さ。
- `ProbabilityVariance`：sample からの class 確率の事後分散。
- `BALD`：予測 label と潜在不確実性の相互情報量。
- `LatentStraddle`：二値のみ。潜在境界 f(x)=0 付近の不確実性。

全 5 基準は現在 q=1 です。確率ベースの共通契約は多 class へ拡張できる構成ですが、`LatentStraddle` は二値の潜在 0 境界に限定します。

## Ensemble

専用 wrapper は不要で、同じ確率空間の契約を使います。entropy と margin は `predict_proba(X)` の予測平均・確率ベクトル、確率分散はメンバー間の変動、BALD は予測 entropy と重み付きメンバー entropy の差を使います。

`ClassificationEnsemblePosterior` の正規化重みを全 4 基準へ引き継ぎます。重みが事後モデル確率なら Bayesian model averaging の重みも反映されます。潜在関数を共有するとは限らない異種 ensemble には `LatentStraddle` を適用しません。

## 較正との合成

`CalibratedBinaryClassifier` は entropy・margin の予測確率を変え、確率 sample も較正してから確率分散・BALD を計算します。`ClassificationProbabilityOfFeasibility` も `predict_proba` を使うため、較正専用の獲得関数 wrapper は不要です。temperature scaling は微分可能で候補の勾配最適化を保ちます。

`LatentStraddle` は変更されない潜在 posterior へ委譲するため、事後確率の較正でスコアは変わりません。較正を潜在 GP 更新と混同しないでください。

NLL・Brier・ECE・MCE を独立の保留データで確認してください。診断は獲得関数ではありません。対応行列・出力形状・制約は[分類契約](../../../docs-internal/contracts/classification-models.md)、利用方法は[分類モデル](../../models/classification.md)を参照してください。

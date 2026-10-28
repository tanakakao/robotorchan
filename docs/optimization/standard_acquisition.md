# 標準の単一目的獲得関数

BoTorch 標準 class を直接使い、重複実装や再公開を行いません。

## 推奨する既定の選択

改善量に基づく BO では、数値的に安定した log-space 版を推奨します。

- `LogExpectedImprovement`：q=1 の解析的期待改善。
- `qLogExpectedImprovement`：Monte Carlo の batch 期待改善。
- `qLogNoisyExpectedImprovement`：ノイズのある観測。

従来の `qExpectedImprovement` には既知の数値問題があり、BoTorch も log 版を推奨します。noisy 改善量でも同じ方針です。再現実験で必要なら従来版を BoTorch から使えますが、既定の推奨ではありません。

その他の標準 API も直接使います。

- `LogProbabilityOfImprovement`：改善確率。
- `UpperConfidenceBound`・`qUpperConfidenceBound`：信頼境界による探索。
- `qSimpleRegret`：posterior utility・simple regret 型の選択。制約なし単一目的の推奨と q-batch 最適化に対応する capability を登録しています。

## 使用例

```python
from botorch.acquisition.logei import qLogExpectedImprovement

acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())
```


返り値は BoTorch 獲得関数であり、`botorch.optim.optimize_acqf` へ直接渡せます。

## モデルとの互換性

互換性は wrapper の有無ではなくモデル・posterior の契約で決まります。標準的な GP モデルは BoTorch 互換の posterior を公開し、`SingleTaskGP` と解析的・MC 獲得関数の統合テストがあります。

non-GP の経験 ensemble は異なります。解析的 Gaussian 獲得関数はその経験 posterior に適用できないため、獲得関数アーキテクチャにある検証 helper を使ってください。

多目的、多忠実度、情報理論、lookahead、Active Learning、level-set は専用ガイドに分けています。[選択の理論](../theory/acquisition/12_selection_guide.md)も参照してください。

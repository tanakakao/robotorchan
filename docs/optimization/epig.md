# Expected Predictive Information Gain

回帰 EPIG は、候補の観測により予測先の入力分布に対する不確実性をどれだけ減らせるかを評価する robotorchan の Active Learning 獲得関数です。

Gaussian 回帰では候補観測と予測対象観測の相互情報量を閉形式で計算できます。有限の target 集合上で平均し、必要なら対象分布の重みを付けます。

```python
from robotorchan.acquisition import ExpectedPredictiveInformationGain

acqf = ExpectedPredictiveInformationGain(
    model,
    target_X,
    observation_noise=0.01,
    target_weights=weights,
)
scores = acqf(candidate_X)
```


現在は q=1 の単一出力 Gaussian 回帰が対象です。文献の分類・ensemble EPIG 推定器とは異なります。batch EPIG には joint conditioning の定義が必要で、独立スコアの近似で代用しません。

target 集合も獲得関数の定義の一部です。運用・テスト分布を代表する集合なら予測重視の選択になりますが、任意のプールを使えば最適化する問い自体が変わります。

理論と関連手法は [Active Learning](../theory/acquisition/09_active_learning.md) を参照してください。

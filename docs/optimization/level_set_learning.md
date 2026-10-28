# Level-set・境界学習

連続応答の level-set 推定を、一般的な回帰 Active Learning と分類から区別します。連続な潜在応答が `target` に達する入力空間の境界を学ぶ課題です。

## Straddle

`Straddle` のスコアは `beta * posterior_std - abs(posterior_mean - target)` です。不確実性が高く、指定水準に近い可能性のある点を優先します。

```python
from robotorchan.acquisition import Straddle

acqf = Straddle(model, target=0.0, beta=1.96)
```


`target` は応答水準、`beta` は不確実性の寄与を表す level-set API の引数であり、分類用語ではありません。

## BoundaryVariance

目標境界付近の事後分散を優先します。近さを事後標準偏差で測るため、目標から遠いことが確かな点は低く評価します。

```python
from robotorchan.acquisition import BoundaryVariance

acqf = BoundaryVariance(model, target=critical_value)
```


`Straddle`、`BoundaryVariance`、`RandomizedStraddle` は現在 q=1 に対応します。多出力 posterior では `output_index` を明示し、構造化 Tensor 出力は事前にスカラー化してください。

## 適用範囲

連続回帰の posterior を対象とし、分類器や分類専用の不確実性基準を追加しません。`RandomizedStraddle` は公開されたランダム化 level-set 獲得関数、`BoundaryVariance` は robotorchan 固有のヒューリスティックです。理論と出典の違いは [Level-set Acquisition](../theory/acquisition/10_level_set.md) を参照してください。

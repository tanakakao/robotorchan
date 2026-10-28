# 情報理論に基づく獲得関数

BoTorch 標準の獲得関数を直接使い、wrapper や再公開を追加しません。

## Max-value Entropy Search

最適値についての情報獲得には BoTorch を直接使います。

```python
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

acqf = qMaxValueEntropy(model=model, candidate_set=candidate_set)
```


`candidate_set` は最大値のサンプリングに使う設計空間の離散化です。選んだ獲得関数が適切な posterior transform に対応する場合を除き、単一出力モデルを使います。

## GIBBON

計算負荷を抑えた下界近似と batch 対応には次を使います。

```python
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

acqf = qLowerBoundMaxValueEntropy(model=model, candidate_set=candidate_set)
```


これらの標準 class は robotorchan の統合テスト対象ですが、alias や wrapper は設けません。

## Randomized Straddle

level-set 推定には `RandomizedStraddle` があります。Inatsu et al. (2024) に従い、手調整の信頼係数を自由度 2 のカイ二乗分布からの抽出に置き換え、Straddle スコアで `sqrt(beta_t)` として使います。

```python
from robotorchan.acquisition import RandomizedStraddle

generator = torch.Generator().manual_seed(0)
acqf = RandomizedStraddle(model, target=0.0, generator=generator)
```


generator を渡すと実験を再現できます。候補選択の 1 ラウンド中は係数を固定して最適化目的を安定させ、次のラウンドで `resample()` を呼びます。

BoTorch に必要な機能がある場合、独自の MES・PES・JES 実装は追加しません。entropy・相互情報量・MES・GIBBON は[情報理論](../theory/acquisition/05_information_theoretic.md)、`RandomizedStraddle` は [level-set 理論](../theory/acquisition/10_level_set.md)を参照してください。

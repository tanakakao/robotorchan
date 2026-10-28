# Thompson sampling と事後分布のサンプリング

有限候補集合からの選択と連続領域上の獲得関数最適化を分けて扱います。

有限プールには `select_thompson_candidates` を使います。事後サンプリングと最大化は BoTorch `MaxPosteriorSampling` に委譲し、候補形状とサンプル数の明示的な契約を追加します。

```python
from robotorchan.acquisition import select_thompson_candidates

selected = select_thompson_candidates(model, choices, num_samples=4)
```


実行可能な領域を列挙済みの離散・混合変数問題に適しています。既定の `replacement=False` では、同じ batch 内で異なる候補行を選びます。

## 獲得関数 wrapper を設けない理由

Thompson sampling は、関数を事後分布から引き、その最大点を選ぶ操作です。BoTorch の既存機能を使い、人工的な獲得関数 class や重複したサンプリング処理を追加しません。

## 連続領域

連続領域ではサンプル関数を bounds 内で最大化する探索戦略も必要です。この最適化を有限プール用 helper に隠しません。trust region や embedding による高次元探索では、異なる探索経路が必要になる場合があります。

## 互換性

BoTorch 互換の posterior が必要です。GP の型判定ではなくモデル・posterior の契約に従います。特別な posterior の意味を持つモデルは、対応を宣言する前に個別の統合テストで確認してください。候補生成のこの経路は MC 獲得関数用 sampling とは異なります。[Posterior sampling](posterior-sampling.md)と[選択の理論](../theory/acquisition/12_selection_guide.md)を参照してください。

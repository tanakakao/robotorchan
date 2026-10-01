# High-dimensional Acquisition Search

## 9.1 問題

入力次元が増えると global candidate search と initialization が難しくなります。

## 9.2 Embedding

REMBO、HeSBO、ALEBO は低次元 embedded space を利用して探索します。

## 9.3 Adaptive subspace

BAxUS は探索の進行に応じて subspace を拡張します。

## 9.4 Model-side reduction との違い

PCA / AE / JointEncoderGP など model 内部の representation と、
candidate-search space の reduction は別です。

詳細は [High-dimensional Search Strategies](../20_high_dimensional_search.md) を参照してください。

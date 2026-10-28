# Dimensionality Reduction GP：座標変換とposterior

## 1. なぜこの実装が必要か

高次元入力をそのままGPでモデル化すると、少数データでは共分散の推定が難しくなる。ここでは `x ∈ R^D` を `z=T(x) ∈ R^d` に変換した後でGPを構築する。数学的な導出と条件は[理論第18章](../theory/18_dimensionality_reduction_gp.md)を正本とする。

**入力を削減することと、候補探索そのものを低次元化することは別である。** `ReducedGP.posterior(X)` は元の入力座標を受け取り内部で変換するので、公開APIの最適化変数が自動的に `d` 次元になるわけではない。

## 2. 変換と学習の処理経路

```text
元の train_X (n × D)  ─→ input_reducer.fit_transform ─→ latent X (n × d)
元の train_Y (n × M)  ─→ output_reducer.fit_transform ─→ latent Y (n × m)
                                                       ↓
                                                 BoTorch SingleTaskGP
                                                       ↓
元の候補 X ─→ input_reducer.transform ─→ latent posterior
                                                       ↓
                                    output_reducer.restore_posterior
                                                       ↓
                                              元の出力空間のposterior
```

`src/robotorchan/models/high_dimensional/reduced/base.py`の`ReductionMixin`は入力の`transform`と出力posteriorの復元を担当する。`ReducedGP`はBoTorch `SingleTaskGP`に削減空間データを渡す。**Delegated**はGPの確率的条件付け、**Implemented / Extended**はreducer接続、座標変換、raw-data保存と出力復元である。

一度fit済みのreducerは再fitせず再利用される。したがって学習データ更新時のreducer再学習・潜在座標の維持は自動的に保証されない。`input_transform`は入力削減後、`outcome_transform`は出力削減後に適用され、元次元と潜在次元を混同して設定してはいけない。

## 3. PCA・PLS・Random Projection

- `PCAGP`: 入力の分散が大きい直交方向を採用する。教師なしであり目的値に重要な低分散方向を落とす可能性がある。
- `PLSGP`: `train_Y`との関係を用いて射影方向を選ぶ。データ分割をまたぐreducer fittingは情報漏洩になる。
- `RandomProjectionGP`: ランダム線形写像を用いる。有限点集合の距離保存に関する性質は、関数の最適解保存を保証しない。
- `OutputPCAGP`、`OutputPLSGP`: 出力も削減する。latent covarianceを元の物理的出力空間へどう復元するかが重要である。

## 4. 最小の利用例

```python
import torch
from robotorchan.models import PCAGP

X = torch.rand(12, 5, dtype=torch.double)
Y = (X[:, :1] - X[:, 1:2]).square()
model = PCAGP(train_X=X, train_Y=Y, n_components=2)
posterior = model.posterior(X[:2])
print(posterior.mean.shape)
```

この例は理論上の処理関係を示すAPI例であり、実行確認済みの証拠ではない。具体的なnamed wrapperの引数は公開実装・Notebookと照合して利用する。

## 5. Mixed入力と制限

カテゴリ列、task/fidelity/contextは安易に連続特徴としてPCA/PLSへ混入させない。Mixed系は設計変数の連続次元だけをreducerへ渡す構造を持つが、reducerの出力順序と`cat_dims`の再配置は個々の実装で確認する。明示的な観測分散`train_Yvar`と出力削減の同時指定は`ReducedGP`で`NotImplementedError`となる。

## 6. 検証と読み進め方

既存[Verification](../../.github/project-records/theory-guide/mapping/verification.md)にPCAの独立射影参照が記録されているが、それはGP予測全体や出力posterior復元の全面的検証ではない。公開座標での予測と固定reducerを通した潜在予測の同値性、出力の平均・共分散復元、dtype/device、異なるデータ分割の学習をpropertyごとに検証する。

[モデル選択](../models/reduced.md)、[Classical reduction Notebook](../../examples/notebooks/19_reduced_gp.ipynb)、[旧Mapping](../../.github/project-records/theory-guide/mapping/models.md)、[Neural representation](19_neural_reduction_gp.md)も参照する。

## 数値 API と利用上の補足

- [Gaussian Process](../theory/02_gaussian_process.md)
- [High-dimensional GP](../theory/08_high_dimensional_gp.md)
- [Neural reduction GP](../theory/19_neural_reduction_gp.md)
- [High-dimensional search](../theory/20_high_dimensional_search.md)
- [MultiTask / MultiOutput](../theory/07_multitask_multioutput.md)
- [Input-reduction model guide](../models/high_dimensional_inputs.md)
- [Output-reduction model guide](../models/high_dimensional_outputs.md)

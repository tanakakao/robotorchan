# Dimensionality Reduction GP：座標変換とposterior

## なぜこの実装が必要か

高次元入力をそのままGPでモデル化すると、少数データでは共分散の推定が難しくなる。ここでは `x ∈ R^D` を `z=T(x) ∈ R^d` に変換した後でGPを構築する。数学的な導出と条件は
[理論第18章](../theory/18_dimensionality_reduction_gp.md)を正本とする。

**入力を削減することと、候補探索そのものを低次元化することは別である。** `ReducedGP.posterior(X)` は元の入力座標を受け取り内部で変換するので、
公開APIの最適化変数が自動的に `d` 次元になるわけではない。

## 変換と学習の処理経路

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

`src/robotorchan/models/high_dimensional/reduced/base.py`の`ReductionMixin`は入力の`transform`と出力
posteriorの復元を担当する。`ReducedGP`はBoTorch `SingleTaskGP`に削減空間データを渡す。**Delegated**はGPの確率的条件付け、
** Implemented / Extended**はreducer接続、座標変換、raw-data保存と出力復元である。

一度fit済みのreducerは再fitせず再利用される。したがって学習データ更新時のreducer再学習・潜在座標の維持は自動的に保証されない。`input_transform`は入力削減後、
`outcome_transform`は出力削減後に適用され、元次元と潜在次元を混同して設定してはいけない。

## PCA・PLS・Random Projection

- `PCAGP`: 入力の分散が大きい直交方向を採用する。教師なしであり目的値に重要な低分散方向を落とす可能性がある。
- `PLSGP`: `train_Y`との関係を用いて射影方向を選ぶ。データ分割をまたぐreducer fittingは情報漏洩になる。
- `RandomProjectionGP`: ランダム線形写像を用いる。有限点集合の距離保存に関する性質は、関数の最適解保存を保証しない。
- `OutputPCAGP`、`OutputPLSGP`: 出力も削減する。latent covarianceを元の物理的出力空間へどう復元するかが重要である。

## Mixed入力と制限

カテゴリ列、task/fidelity/contextは安易に連続特徴としてPCA/PLSへ混入させない。Mixed系は設計変数の連続次元だけをreducerへ渡す構造を持つが、
reducerの出力順序と`cat_dims`の再配置は個々の実装で確認する。
明示的な観測分散`train_Yvar`と出力削減の同時指定は`ReducedGP`で`NotImplementedError`となる。

## 関連する実装

## 関連する実装資料

- [Gaussian Process](../theory/02_gaussian_process.md)
- [High-dimensional GP](../theory/08_high_dimensional_gp.md)
- [Neural reduction GP](../theory/19_neural_reduction_gp.md)
- [High-dimensional search](../theory/20_high_dimensional_search.md)
- [MultiTask / MultiOutput](../theory/07_multitask_multioutput.md)
- [Input-reduction model guide](../models/high_dimensional_inputs.md)
- [Output-reduction model guide](../models/high_dimensional_outputs.md)

## reducerとGPをつなぐメソッド

[reduced/base.py](../../src/robotorchan/models/high_dimensional/reduced/base.py)の
`ReducedGP`は`ReductionMixin`・`ExactGPModelMixin`・BoTorchの`SingleTaskGP`を継承します。
`_prepare_inputs()`・`_prepare_outputs()`がreducerの学習済み状態を調べ、未学習のときだけfitして、潜在データを上流GPのコンストラクタへ渡します。

予測では`posterior()`から`_transform_inputs()`を通し、潜在GPの分布を`_restore_output_posterior()`で元出力へ戻します。
この経路がTheoryの射影と逆写像に対応し、元入力の候補を受け取る境界を保ちます。`condition_on_observations()`も既存の基底を使って新しい観測を変換します。

PCA・PLSの基底計算は[reduction/input.py](../../src/robotorchan/reduction/input.py)、
出力の平均・共分散の復元は[reduction/output.py](../../src/robotorchan/reduction/output.py)が担当します。GPの条件付けを再実装せず、
その前後の座標変換を独自に接続する設計です。

使用手順と結果の可視化は[対応するNotebook](../../examples/notebooks/19_reduced_gp.ipynb)を参照してください。

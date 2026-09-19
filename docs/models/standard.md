# Standard / Mixed / Multi-Fidelity / Variational

## Single-task baseline

`SingleTaskGP` は連続入力の標準的な exact GP baseline です。特殊なデータ構造がない場合、まずこのモデルとの比較を残すと、複雑な surrogate を導入する効果を判断しやすくなります。

`MixedSingleTaskGP` は連続変数とカテゴリ変数が共存する入力空間向けです。カテゴリコードを連続量として距離計算する代替ではなく、categorical covariance を使うことが重要です。

Notebook: [SingleTaskGP](../../examples/notebooks/01_single_task_gp.ipynb)、[MixedSingleTaskGP](../../examples/notebooks/02_mixed_single_task_gp.ipynb)  
Theory: [Gaussian Process](../theory/02_gaussian_process.md)、[Mixed Variables](../theory/05_mixed_variables.md)

## Multi-Fidelity

`SingleTaskMultiFidelityGP` と `MixedSingleTaskMultiFidelityGP` は、評価コストや精度の異なる fidelity 間で情報共有します。シミュレーション解像度、簡易試験と本試験など、fidelity に順序・意味があり相関が期待できる場合に使います。

Mixed版では「カテゴリ入力」と「fidelity」を別の構造として扱います。

Notebook: [Multi-Fidelity](../../examples/notebooks/03_multi_fidelity_gp.ipynb)  
Theory: [Multi-Fidelity](../theory/06_multi_fidelity.md)

## Variational

`SingleTaskVariationalGP` と `MixedSingleTaskVariationalGP` は inducing points を使う近似 GP です。Exact GP の計算量がボトルネックになるデータ量で候補になります。

Exact GP と同じ fitting contract ではありません。`make_mll()` は Variational ELBO 系の学習を前提とするため、データ件数と minibatch の扱いを確認してください。

Notebook: [Variational GP](../../examples/notebooks/06_variational_gp.ipynb)  
Theory: [Variational GP](../theory/09_variational_gp.md)

## 選択基準

カテゴリがあるだけなら Mixed、fidelity 構造があるなら Multi-Fidelity、データ量による exact inference の計算負荷が主問題なら Variational を検討します。これらは排他的なラベルではなく、実装が対応する範囲では複数の問題軸を同時に扱えます。

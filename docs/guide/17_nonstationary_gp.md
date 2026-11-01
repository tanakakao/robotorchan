# Nonstationary GP：理論と実装

## 背景と理論の入口

定常kernelは入力差に依存するが、非定常GPは入力位置により局所lengthscaleや共分散構造が変化する。ARDの次元別固定lengthscaleは異方的でも定常である。
観測ノイズのheteroskedasticityとlatent functionのnonstationarityは別の現象である。

数式と確率的仮定の正本は[対応する理論章](../theory/17_nonstationary_gp.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/robust/nonstationary.py`の`GibbsKernel`は
  `softplus(b+Wx)+lengthscale_floor`による正の局所lengthscaleを使い、位置依存のGibbs共分散を計算する。
- **Extended:** `NonstationarySingleTaskGP`はローカルGibbsKernelをGPyTorchの`ScaleKernel`に収め、
  robotorchan標準`SingleTaskGP`へ渡す。基本GPのposteriorとfitはBoTorch/GPyTorchへ委譲される。
- Mixed/MultiTask/Kronecker版ではカテゴリ軸・task共分散と連続側Gibbs共分散の責務を分ける。

## モデル化と推論上の制限

局所lengthscaleが正であること、固定lengthscaleの場合の定常RBFへの帰着、共分散の対称性・PSDを検査する。入力の狭い観測領域では局所パラメータの識別性が低い可能性があるため、
定常baselineとの予測比較が必要。物理的な相境界が直接判明したと解釈しない。

## Gibbs kernelの局所lengthscaleと実装上の制約

ARDの固定lengthscale ell_jと異なり、GibbsKernelは入力位置ごとにell_j(x)=softplus(b_j+(Wx)_j)+lengthscale_floorを定義する。
2点xとx'に対するcovarianceは

$$
k(x,x')=\prod_j\sqrt{\frac{2\ell_j(x)\ell_j(x')}{\ell_j(x)^2+\ell_j(x')^2}}
\exp\left[-\sum_j\frac{(x_j-x'_j)^2}{\ell_j(x)^2+\ell_j(x')^2}\right].
$$

lengthscale_floor>0が必須で、local_lengthscale(X)は正の値を返す。
全局所lengthscaleが一定ならRBFと同様のstationary covarianceに戻る。ただし入力依存のlengthscaleが存在することは、
事後から実験の物理的境界が一意に同定された証拠ではない。

src/robotorchan/models/robust/nonstationary.pyのNonstationarySingleTaskGPはGibbsKernelをScaleKernelで包み、
SingleTaskGPへcovar_moduleとして渡す。GPのlikelihood、MLL、posteriorはBoTorch/GPyTorchへの委譲を維持する。

MixedNonstationarySingleTaskGPでは連続列をGibbsで、カテゴリ列をCategoricalKernelで扱う。
make_mixed_covar_moduleの加法側と交互作用側には別々のcontinuous Gibbs kernelがあり、local_lengthscale(X)は2組の長さスケールを返す。
したがって1個のlengthscaleベクトルに統合してはならない。連続列が0本ならValueErrorとなる。

NonstationaryMultiTaskGPとKronecker版はtaskの表現が異なり、task列を通常の連続設計座標として扱ってはいけない。
根拠：src/robotorchan/models/robust/nonstationary.py。

## 関連する実装資料

- [Gaussian Process](../theory/02_gaussian_process.md)
- [Kernel](../theory/03_kernel.md)
- [Heteroskedastic / Replicate Noise](../theory/15_heteroskedastic_noise.md)
- [Uncertain-input Gaussian Process](../theory/16_uncertain_input_gp.md)
- [MultiTask / MultiOutput](../theory/07_multitask_multioutput.md)
- [Library-specific nonstationary GP models](../models/nonstationary_gp.md)

## 局所lengthscaleからGibbs共分散へ

[nonstationary.py](../../src/robotorchan/models/robust/nonstationary.py)の
`GibbsKernel`はGPyTorchの`Kernel`を継承し、共分散式をrobotorchan内で実装します。
`local_lengthscale()`は学習可能な切片・傾きによる線形写像へsoftplusと正のfloorを適用し、入力ごと・次元ごとに正のlengthscaleを作ります。

`forward()`は入力対ごとのlengthscaleの二乗和を計算し、TheoryのGibbs kernelの規格化因子を次元方向の積、距離項を和として評価します。
通常経路は入力対の軸をbroadcastで作り、`diag=True`は対応する点対だけを計算します。

`NonstationarySingleTaskGP`は`SingleTaskGP`を継承して、`ScaleKernel(GibbsKernel(...))`を共分散部品として注入します。
局所lengthscaleの写像とGibbs計算は独自実装ですが、Gaussian観測のMLL、条件付け、posteriorは標準GPの上流経路を再利用します。

使用手順と結果の可視化は[対応するNotebook](../../examples/notebooks/18_nonstationary_gp.ipynb)を参照してください。

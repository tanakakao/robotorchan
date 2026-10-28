# Nonstationary GP：理論と実装

## 背景と理論の入口

定常kernelは入力差に依存するが、非定常GPは入力位置により局所lengthscaleや共分散構造が変化する。ARDの次元別固定lengthscaleは異方的でも定常である。観測ノイズのheteroskedasticityとlatent functionのnonstationarityは別の現象である。

数式と確率的仮定の正本は[対応する理論章](../theory/17_nonstationary_gp.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

## 実装アーキテクチャと責務

- **Implemented:** `src/robotorchan/models/robust/nonstationary.py`の`GibbsKernel`は`softplus(b+Wx)+lengthscale_floor`による正の局所lengthscaleを使い、位置依存のGibbs共分散を計算する。
- **Extended:** `NonstationarySingleTaskGP`はローカルGibbsKernelをGPyTorchの`ScaleKernel`に収め、robotorchan標準`SingleTaskGP`へ渡す。基本GPのposteriorとfitはBoTorch/GPyTorchへ委譲される。
- Mixed/MultiTask/Kronecker版ではカテゴリ軸・task共分散と連続側Gibbs共分散の責務を分ける。

## APIと実装を読むための最小例

```python
import torch
from robotorchan.models import NonstationarySingleTaskGP

X = torch.linspace(0, 1, 6, dtype=torch.double).unsqueeze(-1)
Y = torch.sin(6 * X)
model = NonstationarySingleTaskGP(train_X=X, train_Y=Y)
print(model.gibbs_kernel.local_lengthscale(X).shape)
```

コード例は公開APIを示す静的な例であり、このセッションで実行して成功したとの主張ではない。

## 適用条件、テスト、限界

局所lengthscaleが正であること、固定lengthscaleの場合の定常RBFへの帰着、共分散の対称性・PSDを検査する。入力の狭い観測領域では局所パラメータの識別性が低い可能性があるため、定常baselineとの予測比較が必要。物理的な相境界が直接判明したと解釈しない。

旧[モデルMapping](../../.github/project-records/theory-guide/mapping/models.md)・[数学的検証Mapping](../../.github/project-records/theory-guide/mapping/verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../models/robust_noise.md)には利用者向けの比較を記す。

## Gibbs kernelの局所lengthscaleと実装上の制約

ARDの固定lengthscale ell_jと異なり、GibbsKernelは入力位置ごとにell_j(x)=softplus(b_j+(Wx)_j)+lengthscale_floorを定義する。2点xとx'に対するcovarianceは

$$
k(x,x')=\prod_j\sqrt{\frac{2\ell_j(x)\ell_j(x')}{\ell_j(x)^2+\ell_j(x')^2}}
\exp\left[-\sum_j\frac{(x_j-x'_j)^2}{\ell_j(x)^2+\ell_j(x')^2}\right].
$$

lengthscale_floor>0が必須で、local_lengthscale(X)は正の値を返す。全局所lengthscaleが一定ならRBFと同様のstationary covarianceに戻る。ただし入力依存のlengthscaleが存在することは、事後から実験の物理的境界が一意に同定された証拠ではない。

src/robotorchan/models/robust/nonstationary.pyのNonstationarySingleTaskGPはGibbsKernelをScaleKernelで包み、SingleTaskGPへcovar_moduleとして渡す。GPのlikelihood、MLL、posteriorはBoTorch/GPyTorchへの委譲を維持する。

MixedNonstationarySingleTaskGPでは連続列をGibbsで、カテゴリ列をCategoricalKernelで扱う。make_mixed_covar_moduleの加法側と交互作用側には別々のcontinuous Gibbs kernelがあり、local_lengthscale(X)は2組の長さスケールを返す。したがって1個のlengthscaleベクトルに統合してはならない。連続列が0本ならValueErrorとなる。

NonstationaryMultiTaskGPとKronecker版はtaskの表現が異なり、task列を通常の連続設計座標として扱ってはいけない。根拠：src/robotorchan/models/robust/nonstationary.py。ここでの説明はソース契約に基づき、PSDや定常極限の独立数値テスト、定常kernelとの予測性能比較は未実施。

## 数値 API と利用上の補足

- [Gaussian Process](../theory/02_gaussian_process.md)
- [Kernel](../theory/03_kernel.md)
- [Heteroskedastic / Replicate Noise](../theory/15_heteroskedastic_noise.md)
- [Uncertain-input Gaussian Process](../theory/16_uncertain_input_gp.md)
- [MultiTask / MultiOutput](../theory/07_multitask_multioutput.md)
- [Library-specific nonstationary GP models](../models/nonstationary_gp.md)

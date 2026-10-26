# Nonstationary GP：理論と実装

## 背景と理論の入口

定常kernelは入力差に依存するが、非定常GPは入力位置により局所lengthscaleや共分散構造が変化する。ARDの次元別固定lengthscaleは異方的でも定常である。観測ノイズのheteroskedasticityとlatent functionのnonstationarityは別の現象である。

数式と確率的仮定の正本は[対応する理論章](../../../theory/17_nonstationary_gp.md)を参照。本章ではrobotorchanの実装責務・推論・制約に重点を置く。

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

旧[モデルMapping](../models.md)・[数学的検証Mapping](../verification.md)を残し、学習ガイド本文に過去のPR/CI履歴を混入させない。[モデル選択ガイド](../../../models/robust_noise.md)には利用者向けの比較を記す。

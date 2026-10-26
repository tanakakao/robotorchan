# Neural Representation Learning for GP：FrozenとJointの違い

## 1. 数理モデルの入口

線形射影 `z=Wᵀx` に代えてニューラルencoder `z=h_θ(x)` を用いる。**encoderがいつGPの学習目的を受け取るか**がモデルの本質的な違いである。[理論第19章](../../../theory/19_neural_reduction_gp.md)が数理上の正本となる。

| モデル群 | 表現を更新する損失 | GPとの関係 |
| --- | --- | --- |
| AutoEncoderGP / VAEGP | 再構成・KLなど | 先にreducerを学習して固定 |
| SupervisedAutoEncoderGP / SupervisedVAEGP | 表現学習に教師Yを利用 | 事前のsupervised fittingとGP学習を分離 |
| JointEncoderGP / HybridAutoEncoderGP / JointVAEGP | GP objectiveをencoderへ伝播 | encoderとGPをjoint training |

Supervised representationとjoint fittingは異なる。VAEモデルのposterior-mean latent representationを使う場合、VAEのlatent sampling uncertaintyがGP posteriorへ自動的に積分されたことにはならない。

## 2. 実装と処理経路

Frozen reducerは`src/robotorchan/models/high_dimensional/reduced/base.py`と`src/robotorchan/reduction/neural.py`を中心に構成され、reducerがfit済みなら同じ座標変換を再利用する。基礎GPのposteriorはBoTorchに**Delegated**、入力のencoder変換はrobotorchanの**Implemented / Extended**である。

Joint系の`JointEncoderGP`は`src/robotorchan/models/high_dimensional/reduced/joint_neural.py`にあり、正規化した元入力からencoderを通して潜在入力を作る。`ExactGPModelMixin`とBoTorch `SingleTaskGP`を基礎とするが、encoderをGP objectiveで更新する専用の`training_loss`契約がある。単に標準GPの`fit_gpytorch_mll(model.make_mll())`を呼べば常にencoderまでjointに更新される、とは主張しない。

```text
Frozen: X → encoder.fit → freeze → latent X → GP MLL → posterior
Joint:  X → encoder_θ(X) → GP likelihood / training_loss
                 ↑                    │
                 └──── gradient ──────┘
```

## 3. 構築例と学習の注意点

```python
import torch
from robotorchan.models import JointEncoderGP

X = torch.rand(16, 6, dtype=torch.double)
Y = (X[:, :1] - 0.3).square()
model = JointEncoderGP(
    train_X=X,
    train_Y=Y,
    latent_dim=2,
    hidden_dims=(16, 8),
)
print(model.latent_dim)
```

ここではモデルの構築契約だけを示し、学習済み・posterior検証済みとは扱わない。Joint系ではGPとencoder双方のパラメータが適切にoptimizerへ渡ること、勾配がencoderへ実際に到達すること、評価時に同じ入力正規化を行うことを確認する必要がある。

## 4. どのモデルを使うか

AEは入力分布の再構成、supervised reducerはYに関連する特徴、joint GPは予測目的に適合した表現を優先する。一方で、joint trainingは小標本の過学習・最適化不安定性・潜在空間の同定困難性を伴う。Mixed入力ではカテゴリを連続encoderに一律投入しない。モデルごとにnative categorical処理、one-hot、連続部のみの削減を確認する。

## 5. 検証境界と資料

独立参照で確認すべきものは、(1) frozen encoder再fitが起きないこと、(2) training_lossからencoderへのgradient、(3) train/predict transformの一貫性、(4) joint posteriorの出力形状・分散、(5) Mixedカテゴリ列の保持である。CI成功やモデル作成だけで表現学習が適切に収束した証明とはならない。

参照：[Neural reduction Notebook](../../../../examples/notebooks/20_neural_reduction_gp.ipynb)、[モデルガイド](../../../models/reduced.md)、[旧Mapping](../models.md)、[Verification](../verification.md)、[Dimensionality Reduction](18_dimensionality_reduction_gp.md)。

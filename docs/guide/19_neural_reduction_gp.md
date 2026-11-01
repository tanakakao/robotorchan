# Neural Representation Learning for GP：FrozenとJointの違い

## 1. 数理モデルの入口

線形射影 `z=Wᵀx` に代えてニューラルencoder `z=h_θ(x)` を用いる。**encoderがいつGPの学習目的を受け取るか**がモデルの本質的な違いである。[理論第19章](../theory/19_neural_reduction_gp.md)が数理上の正本となる。

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

参照：[Neural reduction Notebook](../../examples/notebooks/20_neural_reduction_gp.ipynb)、[モデルガイド](../models/reduced.md)、[旧Mapping](../../.github/project-records/theory-guide/mapping/models.md)、[Verification](../../.github/project-records/theory-guide/mapping/verification.md)、[Dimensionality Reduction](18_dimensionality_reduction_gp.md)。

## 6. Frozenとjointの公開モデル

| 項目 | Frozen AE / VAE | Joint / Hybrid |
| --- | --- | --- |
| Encoderの学習 | GP構築前のreducer fitting | GPの予測目的と同時 |
| GPからencoderへの勾配 | なし | あり |
| GP学習中の潜在基底 | 固定 | 変化する |
| 学習経路 | reducer fitとGP fit | `training_loss()`または対応する`fit()` |
| Reconstruction regularization | reducer側 | Hybrid等のjoint objective |

`AutoEncoderGP`は事前学習したencoderを固定してGPを構築する。
`VAEGP`もfrozen reducerで、通常の`posterior(X)`にはlatent posterior meanを使う。
`SupervisedAutoEncoderGP`はreducer学習にoutcome-prediction lossを加え、
`SupervisedVAEGP`はVAE objectiveへoutcome-aware supervisionを加える。
このsupervised reducerも、GP MLLとencoderを共同学習するモデルではない。

`JointEncoderGP`はencoderをpredictive GPの一部として学習する。
`HybridAutoEncoderGP`はdecoderを加え、予測目的とreconstructionを組み合わせる。
`JointVAEGP`はprobabilistic encoderとGP predictive objectiveを共同で扱う。
joint auxiliary lossは標準MLL helperだけでは表せないため、
[Joint neural training](../models/joint_neural_training.md)の学習契約に従う。

## 7. VAEのlatent uncertainty

latent変数を平均だけに置き換えると、表現の不確実性を失う場合がある。
`JointVAEGP.uncertainty_aware_posterior()`はlatent sampleごとのGP予測から
Monte Carloでpredictive momentsを求め、Gaussian posteriorへmoment matchingする。
各sampleのposteriorを保持した一般の非Gaussian mixtureを返すAPIとは異なる。

frozen reducerでもjoint modelでも、公開posteriorは元空間の候補`X`を受けて内部でencodeする。
frozen familyは`ReducedGP`の座標変換・posterior復元の基盤を利用する。
encoderの存在だけでacquisitionの互換性を判断せず、posterior typeとshapeを確認する。

## 8. Mixed・MultiTask・Kroneckerで保つ構造

Mixedのfrozen familyには`MixedAutoEncoderGP`、`MixedSupervisedAutoEncoderGP`、
`MixedVAEGP`、`MixedSupervisedVAEGP`がある。
joint側には`MixedJointEncoderGP`、`MixedHybridAutoEncoderGP`があり、
カテゴリ列をcontinuous encoderから分離する。
Mixed Hybridのreconstruction lossもcontinuous columnsを対象とする。

| joint family | Long-format MultiTask | Kronecker MultiTask |
| --- | --- | --- |
| Encoder | `JointEncoderMultiTaskGP` | `JointEncoderKroneckerMultiTaskGP` |
| Hybrid AE | `HybridAutoEncoderMultiTaskGP` | `HybridAutoEncoderKroneckerMultiTaskGP` |
| VAE | `JointVAEMultiTaskGP` | `JointVAEKroneckerMultiTaskGP` |

long formatでは`task_feature`をencoderから除外し、data featuresの潜在表現と
タスク識別子を再結合する。Kronecker形式ではtask identityがYの軸なので、
Xのdata featuresをencodeする。
`MixedJointEncoderMultiTaskGP`はcategorical design columnsとtask featureの両方を分離する。
これらの名前からMixed × joint × MultiTaskの全直積への対応を推測しない。

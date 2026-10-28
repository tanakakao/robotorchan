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

## 理論章から分離した実装上の条件

### 14. 理論と実装の境界

Neural reductionでは、encoderの事前学習とGPとの共同学習を区別します。
また、潜在変数の不確実性を点推定に置き換える場合、予測分布の分散が過小評価される可能性があります。
Mixed入力では連続変数の表現学習とカテゴリ共分散を区別し、Multi-task入力ではタスク構造を適切に保持します。

モデルごとのAPI、学習手順、事後分布の呼び出し方は[高次元入力モデル](../models/high_dimensional_inputs.md)を参照してください。

#### 2. VAE GP：実装上の補足

したがって通常の `VAEGP.posterior(X)` は

#### 7. Joint VAE：実装上の補足

joint model の学習では `training_loss()` を使います。

#### 現行MultiTask / Kroneckerのjoint family：実装上の補足

#### 現行MultiTask / Kroneckerのjoint family

#### 11. Frozen reducer と joint model の比較：実装上の補足

| 項目 | Frozen AE/VAE | Joint / Hybrid |
| --- | --- | --- |
| Encoder fitting | GP構築前 | GPと同時 |
| GP objectiveからencoderへの勾配 | なし | あり |
| Latent basis | GP fitting中は固定 | training中に変化 |
| 学習契約 | reducer fit + GP fit | `training_loss()` |
| Reconstruction regularization | reducer側 | Hybrid等でjoint |
| 実装複雑性 | 比較的低い | 高い |

#### 12. BoTorch compatibility：実装上の補足

### 12. BoTorch compatibility

#### 12. BoTorch compatibility：実装上の補足

重要なのは、encoder が存在しても acquisition function から見た model interface は
BoTorch posterior contract を維持することです。

#### 12. BoTorch compatibility：実装上の補足

一方で joint auxiliary loss は標準 MLL fit helper だけでは表現できないため、
training loop は `training_loss()` contract に従います。

## 数値 API と利用上の補足

`Joint VAE GP.uncertainty_aware_posterior()` はこの考え方を Monte Carlo 近似し、
latent-sample posterior mixture の moment を近似します。

さらに `uncertainty_aware_posterior()` はlatent-sampleごとのGP posteriorをそのまま返すAPIではなく、
Monte Carloで得たpredictive momentsをmoment matchingしてGaussian posteriorとして返します。
したがって一般の非Gaussian mixture posteriorを完全に保持する操作とは区別します。

現在はjoint representationについても

| 項目 | Frozen AE/VAE | Joint / Hybrid |
| --- | --- | --- |
| Encoder fitting | GP構築前 | GPと同時 |
| GP objectiveからencoderへの勾配 | なし | あり |
| Latent basis | GP fitting中は固定 | training中に変化 |
| 学習契約 | reducer fit + GP fit | `training_loss()` |
| Reconstruction regularization | reducer側 | Hybrid等でjoint |
| 実装複雑性 | 比較的低い | 高い |

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

`AutoEncoderGP` は encoder を reducer として先に学習し、その latent coordinate を固定して
GP を構築します。

```text
train_X
 -> fit AE once
 -> frozen encoder
 -> latent train_X
 -> GP
```


この点が `JointEncoderGP` との本質的な違いです。

`VAEGP` は frozen VAE の posterior-mean latent representation を GP input として使います。

`SupervisedAutoEncoderGP` は reducer の事前学習時に reconstruction loss に加えて
outcome-prediction loss を使います。

`SupervisedVAEGP` は VAE objective に outcome-aware supervision を追加します。

通常の `SupervisedVAEGP` もfrozen reducer modelであり、GP MLLとencoderをjoint training
するモデルではありません。

## 5. JointEncoderGP

`JointEncoderGP` は encoder を predictive GP model の一部として扱います。

## 6. HybridAutoEncoderGP

`HybridAutoEncoderGP` は decoder を追加し、

`JointVAEGP` 系では probabilistic encoder と GP predictive objective を同時に扱います。

重要なのは、joint VAE が単なる `VAEGP` の別名ではないことです。

`VAEGP` は fitted reducer を固定して GP を構築するのに対し、joint VAE では encoder が
predictive training の一部です。

`JointVAEGP.uncertainty_aware_posterior()` はこの考え方を Monte Carlo 近似し、
latent-sample posterior mixture の moment を近似します。

- `MixedAutoEncoderGP`
- `MixedSupervisedAutoEncoderGP`
- `MixedVAEGP`
- `MixedSupervisedVAEGP`
- `MixedJointEncoderGP`
- `MixedHybridAutoEncoderGP`

Mixed joint modelではcategoryをencoderへ入れないだけでなく、
`MixedHybridAutoEncoderGP` のreconstruction lossもcontinuous columnsを対象にします。

- `JointEncoderMultiTaskGP`
- `JointEncoderKroneckerMultiTaskGP`
- `HybridAutoEncoderMultiTaskGP`
- `HybridAutoEncoderKroneckerMultiTaskGP`
- `JointVAEMultiTaskGP`
- `JointVAEKroneckerMultiTaskGP`

long-format MultiTaskではtask featureをencoderから除外し、data featuresだけをlatentへ写像してから
task identityを再結合します。Kronecker形式ではtask identityがXの列ではなくY側のtask axisなので、
train_Xのdata featuresをencoderへ渡せます。

また `MixedJointEncoderMultiTaskGP` ではcategorical design columnsとtask featureの両方を
continuous encoderから分離します。モデル名の存在から他のMixed × joint × MultiTaskの全直積まで
自動的に実装済みと推測しないことが重要です。

frozen neural reducer は `ReducedGP` infrastructure を使うため、public posterior は
original-space candidate `X` を受け、内部で encode します。

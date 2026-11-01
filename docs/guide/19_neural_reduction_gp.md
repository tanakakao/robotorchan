# Neural Representation Learning for GP：FrozenとJointの違い

## 数理モデルの入口

線形射影 `z=Wᵀx` に代えてニューラルencoder `z=h_θ(x)` を用いる。**encoderがいつGPの学習目的を受け取るか**がモデルの本質的な違いである。
[理論第19章](../theory/19_neural_reduction_gp.md)が数理上の正本となる。

- **AutoEncoderGP / VAEGP**
  表現を更新する損失：再構成・KLなど
  GPとの関係：先にreducerを学習して固定

- **SupervisedAutoEncoderGP / SupervisedVAEGP**
  表現を更新する損失：表現学習に教師Yを利用
  GPとの関係：事前のsupervised fittingとGP学習を分離

- **JointEncoderGP / HybridAutoEncoderGP / JointVAEGP**
  表現を更新する損失：GP objectiveをencoderへ伝播
  GPとの関係：encoderとGPをjoint training

Supervised representationとjoint fittingは異なる。VAEモデルのposterior-mean latent representationを使う場合、
VAEのlatent sampling uncertaintyがGP posteriorへ自動的に積分されたことにはならない。

## 実装と処理経路

Frozen reducerは`src/robotorchan/models/high_dimensional/reduced/base.py`と
`src/robotorchan/reduction/neural.py`を中心に構成され、reducerがfit済みなら同じ座標変換を再利用する。
基礎GPのposteriorはBoTorchに** Delegated**、入力のencoder変換はrobotorchanの**Implemented / Extended**である。

Joint系の`JointEncoderGP`は`src/robotorchan/models/high_dimensional/reduced/joint_neural.py`にあり、
正規化した元入力からencoderを通して潜在入力を作る。`ExactGPModelMixin`とBoTorch `SingleTaskGP`を基礎とするが、
encoderをGP objectiveで更新する専用の`training_loss`契約がある。
単に標準GPの`fit_gpytorch_mll(model.make_mll())`を呼べば常にencoderまでjointに更新される、とは主張しない。

```text
Frozen: X → encoder.fit → freeze → latent X → GP MLL → posterior
Joint:  X → encoder_θ(X) → GP likelihood / training_loss
                 ↑                    │
                 └──── gradient ──────┘
```

## どのモデルを使うか

AEは入力分布の再構成、supervised reducerはYに関連する特徴、joint GPは予測目的に適合した表現を優先する。一方で、
joint trainingは小標本の過学習・最適化不安定性・潜在空間の同定困難性を伴う。Mixed入力ではカテゴリを連続encoderに一律投入しない。
モデルごとにnative categorical処理、one-hot、連続部のみの削減を確認する。

## Frozenとjointの公開モデル

| 項目 | Frozen AE / VAE | Joint / Hybrid |
| --- | --- | --- |
| Encoderの学習 | GP構築前のreducer fitting | GPの予測目的と同時 |
| GPからencoderへの勾配 | なし | あり |
| GP学習中の潜在基底 | 固定 | 変化する |
| 学習経路 | reducer fitとGP fit | `training_loss()`または対応する`fit()` |
| Reconstruction regularization | reducer側 | Hybrid等のjoint objective |

`AutoEncoderGP`は事前学習したencoderを固定してGPを構築する。`VAEGP`もfrozen reducerで、
通常の`posterior(X)`にはlatent posterior meanを使う。
`SupervisedAutoEncoderGP`はreducer学習にoutcome-prediction lossを加え、
`SupervisedVAEGP`はVAE objectiveへoutcome-aware supervisionを加える。このsupervised reducerも、
GP MLLとencoderを共同学習するモデルではない。

`JointEncoderGP`はencoderをpredictive GPの一部として学習する。`HybridAutoEncoderGP`はdecoderを加え、
予測目的とreconstructionを組み合わせる。`JointVAEGP`はprobabilistic encoderとGP predictive objectiveを共同で扱う。
joint auxiliary lossは標準MLL helperだけでは表せないため、
[Joint neural training](../models/joint_neural_training.md)の学習契約に従う。

## VAEのlatent uncertainty

latent変数を平均だけに置き換えると、表現の不確実性を失う場合がある。
`JointVAEGP.uncertainty_aware_posterior()`はlatent sampleごとのGP予測からMonte Carloでpredictive momentsを求め、
Gaussian posteriorへmoment matchingする。各sampleのposteriorを保持した一般の非Gaussian mixtureを返すAPIとは異なる。

frozen reducerでもjoint modelでも、公開posteriorは元空間の候補`X`を受けて内部でencodeする。
frozen familyは`ReducedGP`の座標変換・posterior復元の基盤を利用する。encoderの存在だけでacquisitionの互換性を判断せず、
posterior typeとshapeを確認する。

## Mixed・MultiTask・Kroneckerで保つ構造

Mixedのfrozen familyには`MixedAutoEncoderGP`、`MixedSupervisedAutoEncoderGP`、`MixedVAEGP`、
`MixedSupervisedVAEGP`がある。joint側には`MixedJointEncoderGP`、`MixedHybridAutoEncoderGP`があり、
カテゴリ列をcontinuous encoderから分離する。Mixed Hybridのreconstruction lossもcontinuous columnsを対象とする。

| joint family | Long-format MultiTask | Kronecker MultiTask |
| --- | --- | --- |
| Encoder | `JointEncoderMultiTaskGP` | `JointEncoderKroneckerMultiTaskGP` |
| Hybrid AE | `HybridAutoEncoderMultiTaskGP` | `HybridAutoEncoderKroneckerMultiTaskGP` |
| VAE | `JointVAEMultiTaskGP` | `JointVAEKroneckerMultiTaskGP` |

long formatでは`task_feature`をencoderから除外し、data featuresの潜在表現とタスク識別子を再結合する。
Kronecker形式ではtask identityがYの軸なので、Xのdata featuresをencodeする。
`MixedJointEncoderMultiTaskGP`はcategorical design columnsとtask featureの両方を分離する。
これらの名前からMixed × joint × MultiTaskの全直積への対応を推測しない。

## encoderの勾配がGPへ接続される場所

[joint_neural.py](../../src/robotorchan/models/high_dimensional/reduced/joint_neural.py)は
`JointEncoderGP`とHybrid AEの入力変換・損失を定義します。encoderがモデルのsubmoduleとしてパラメータを保持し、`forward()`へ届く特徴表現を作るため、
予測目的の勾配がencoderまで戻ります。GPのGaussian推論は上流部品を使い、変換と補助損失の結合をローカルに定義する構造です。

[joint_vae.py](../../src/robotorchan/models/high_dimensional/reduced/joint_vae.py)では
`training_loss()`が予測目的とVAE側の項を組み合わせます。`uncertainty_aware_posterior()`はlatent sampleごとの予測momentを集計し、
平均の変動も含めてGaussianへmoment matchingします。通常の潜在平均だけによる予測と、この周辺化の近似は別メソッドです。

使用手順と結果の可視化は[対応するNotebook](../../examples/notebooks/20_neural_reduction_gp.ipynb)を参照してください。

# Expressive GP：表現力の源と推論の違い

## 1. どの構造を柔軟にするか

Expressive GPは単一のアルゴリズムではない。[理論第22章](../../../theory/22_expressive_gp.md)では、deterministic neural features、stochastic GP hierarchy、neural-network-derived kernel prior、spectral covarianceの違いを説明している。この章では、それぞれの**posteriorの意味**と学習APIへの接続を扱う。

| Family | 数理構造 | robotorchanでの公開入口 | 学習・推論の要点 |
| --- | --- | --- | --- |
| Deep Kernel Learning (DKL) | `k(g_θ(x),g_θ(x'))` | `JointEncoderGP` | encoderとExact GPをjoint trainingする |
| DeepGP | 階層的な確率GP写像 | `SingleTaskDeepGP` | variational / MC推論でありExact MLLではない |
| Infinite-width BNN | 無限幅NNに対応するNNGP kernel | `InfiniteWidthBNNGP` | kernel prior上のExact GP |
| Spectral Mixture | 複数周波数成分による定常kernel | `SpectralMixtureGP` | Exact GP、mixture数と初期化に敏感 |

## 2. アルゴリズムから実装へ

**Extended / Delegated：** DKLに関連する`src/robotorchan/models/high_dimensional/reduced/joint_neural.py`の`JointEncoderGP`は、元入力をencoderへ通してGPに渡す。BoTorch/GPyTorchが基本GPの推論を扱う一方、robotorchanはfeature extractorとGP objectiveを結合する`training_loss`経路を担う。これはfrozen AEの「先に表現を学習してから固定する」経路と区別する。

**Implemented / Delegated：** DeepGP系は確率的な複数GP layerを通すため、出力のサンプリングとvariational学習を要求する。posterior samplesの意味がDKLのdeterministic latent featuresと異なる。NNGPおよびSpectral Mixtureはkernel側の選択なので、DeepGPの層構造を持たない。

**Mixed / MultiTask契約：** `MixedJointEncoderGP`ではカテゴリ列を連続encoderへ無条件に渡さない。`JointEncoderMultiTaskGP`のlong-format task featureと`JointEncoderKroneckerMultiTaskGP`のblock-design task軸は別であり、予測時に与える元入力形状も異なる。実際の対応モデルは[Expressiveモデルガイド](../../../models/expressive.md)で区別する。

## 3. 最小の入口例

```python
import torch
from robotorchan.models import JointEncoderGP

train_X = torch.rand(16, 4, dtype=torch.double)
train_Y = (train_X[:, :1] - 0.4).square()
model = JointEncoderGP(
    train_X=train_X,
    train_Y=train_Y,
    latent_dim=2,
    hidden_dims=(16, 8),
)
print(model.latent_dim)
```

モデル構築例であり、学習完了やencoder勾配検証を示すものではない。Joint系の実用では、`training_loss`に基づくoptimizer step、encoder gradient、保存・再読込後の入力変換、posteriorサンプルshapeを確認する。

## 4. 互換性と検証の境界

MC acquisitionがposteriorsを消費できても、全Expressive familyで任意の`q`、constraint、`optimize_acqf`が成功するとは限らない。DeepGPのstochastic posteriorを単一Gaussian posteriorとみなさず、NNGPのpriorを有限NNのweight posteriorとみなさない。Spectral Mixtureでは入力スケール・初期化・mixture数を比較条件として固定する。

参照：[Neural Representation](19_neural_reduction_gp.md)、[Expressive Notebook](../../../../examples/notebooks/24_expressive_surrogate_gp.ipynb)、[旧Models Mapping](../models.md)、[Verification](../verification.md)。完了した検証の範囲は数値propertyと実行環境により限定する。

# Neural Representation Learning for GP

## 直感

PCA/PLS が線形射影なのに対し、neural reducer は

```text
z = encoder_theta(x)
```

という非線形表現を学習します。robotorchan では「表現を先に学習して固定する方式」と
「GP objective と encoder を共同学習する方式」を区別します。

## AutoEncoder

AutoEncoder は

```text
x -> encoder -> z -> decoder -> x_hat
```

として reconstruction loss を最小化します。`AutoEncoderGP` は事前学習した encoder を
freeze して GP に接続します。

## VAE

VAE は latent distribution `q(z|x)` を学習し、reconstruction と KL regularization を使います。
通常の `VAEGP` は posterior mean latent code を決定論的な GP input として使います。

## Supervised representation

`SupervisedAutoEncoderGP` / `SupervisedVAEGP` は reconstruction だけでなく Y に関連する
supervised loss を加えます。入力構造の再現より予測に有用な latent space を得る狙いがあります。

## Joint learning

`JointEncoderGP` は GP marginal likelihood の勾配を encoder まで伝えます。
`HybridAutoEncoderGP` は GP objective と reconstruction regularization を併用し、
`JointVAEGP` はさらに VAE の probabilistic regularization を組み込みます。

joint model は単純な `fit_gpytorch_mll(model.make_mll())` だけでは auxiliary loss を含められない
ため、robotorchan では `training_loss()` を学習契約とします。

## Latent uncertainty

VAE の `q(z|x)` を平均値だけに潰すと representation uncertainty は posterior に伝わりません。
`JointVAEGP.uncertainty_aware_posterior()` は latent sample に対する GP posterior mixture の
モーメントを Monte Carlo 近似します。

## MultiTask / Mixed

MultiTask では task identity を reducer から分離し、Mixed input では categorical feature を
continuous neural encoder に暗黙に混ぜません。named class の全直積は作らず、共通 reduction
base で構成する場合があります。

## 使い分け

- 小標本・線形構造: PCA/PLS を先に比較
- 非線形 manifold: AE/VAE
- Y に関連する表現: supervised reducer
- GP predictive objective まで含めたい: Joint/Hybrid
- representation uncertainty も必要: Joint VAE の uncertainty-aware posterior

詳細は [high-dimensional inputs](../models/high_dimensional_inputs.md) を参照してください。

# Neural Representation Learning for Gaussian Processes

## 直感

線形 reduction では

```text
z = W^T x
```

のような固定された線形 subspace を使います。

neural representation では

```text
z = encoder_theta(x)
```

として非線形な latent representation を学習します。

robotorchan では neural reduction を大きく2種類に分けます。

1. encoder を事前学習して固定し、その後 GP を学習する frozen-reducer model
2. GP objective の勾配を encoder まで伝える jointly-trained model

この区別は学習 objective、posterior semantics、更新方法のすべてに影響します。

## 1. Frozen AutoEncoder GP

AutoEncoder は

```text
x
 -> encoder
 -> z
 -> decoder
 -> x_hat
```

として reconstruction loss を最小化します。

典型的には

```text
L_AE = mean ||x - x_hat||^2
```

です。

`AutoEncoderGP` は encoder を reducer として先に学習し、その latent coordinate を固定して
GP を構築します。

```text
train_X
 -> fit AE once
 -> frozen encoder
 -> latent train_X
 -> GP
```

GP marginal likelihood の勾配は、この事前学習済み encoder の表現学習には戻りません。

この点が `JointEncoderGP` との本質的な違いです。

## 2. VAE GP

Variational AutoEncoder は deterministic code ではなく latent distribution を学習します。

```text
q_phi(z | x) = N(mu_phi(x), diag(sigma_phi(x)^2))
```

典型的な VAE objective は

```text
L_VAE
  = reconstruction loss
  + beta * KL(
      q_phi(z | x)
      || p(z)
    )
```

です。

`VAEGP` は frozen VAE の posterior-mean latent representation を GP input として使います。

したがって通常の `VAEGP.posterior(X)` は

```text
X
 -> E[z | X]
 -> GP posterior
```

という deterministic latent-input approximation です。

VAE を使っているだけで representation uncertainty が GP posterior へ自動的に伝播するわけでは
ありません。

## 3. Supervised AutoEncoder

reconstruction に最適な latent direction と、`Y` の予測に重要な direction は一致しないことが
あります。

`SupervisedAutoEncoderGP` は reducer の事前学習時に reconstruction loss に加えて
outcome-prediction loss を使います。

概念的には

```text
L
  = L_reconstruction
  + lambda_y * L_supervised
```

です。

これにより input manifold の再構成だけでなく、surrogate prediction に有用な representation を
残すことを狙います。

ただし reducer が `Y` を利用するため、validation では representation fitting を含めた
data leakage に注意する必要があります。

## 4. Supervised VAE

`SupervisedVAEGP` は VAE objective に outcome-aware supervision を追加します。

概念的には

```text
L
  = L_reconstruction
  + beta * L_KL
  + lambda_y * L_supervised
```

です。

`beta` は latent prior regularization の強さ、`supervised_weight` は outcome-aware loss の
重みを制御します。

通常の `SupervisedVAEGP` も frozen reducer model であり、GP MLL と encoder を joint training
するモデルではありません。

## 5. JointEncoderGP

`JointEncoderGP` は encoder を predictive GP model の一部として扱います。

```text
X
 -> encoder_theta(X)
 -> GP
 -> marginal likelihood
```

GP marginal likelihood の勾配が `theta` まで伝わるため、latent representation は
GP predictive objective によって更新されます。

基本 training objective は

```text
L_joint = - log p(Y | encoder_theta(X))
```

です。

robotorchan では `training_loss()` を joint neural GP の共通学習契約とします。

これは frozen `ReducedGP` と異なり、encoder parameter が GP training loop の一部だからです。

## 6. HybridAutoEncoderGP

GP MLL のみで representation を学習すると、予測に有利でも input structure を強く潰した
latent space になる可能性があります。

`HybridAutoEncoderGP` は decoder を追加し、

```text
L_hybrid
  = - log p(Y | encoder_theta(X))
  + lambda_rec * L_reconstruction
```

を使います。

`reconstruction_weight = 0` なら reconstruction regularization を無効化できます。

reconstruction は standardized input space で計算されます。

## 7. Joint VAE

`JointVAEGP` 系では probabilistic encoder と GP predictive objective を同時に扱います。

VAE の

- reconstruction
- KL regularization
- stochastic latent representation

と GP likelihood / marginal objective を組み合わせます。

重要なのは、joint VAE が単なる `VAEGP` の別名ではないことです。

`VAEGP` は fitted reducer を固定して GP を構築するのに対し、joint VAE では encoder が
predictive training の一部です。

joint model の学習では `training_loss()` を使います。

## 8. Latent uncertainty

VAE encoder が

```text
q(z | x)
```

を返しても、posterior mean `mu(x)` だけを GP input にすれば latent uncertainty は失われます。

representation uncertainty を反映するには、例えば

```text
z_s ~ q(z | x)

p(f | x, D)
  ~= mixture_s p(f | z_s, D)
```

のように latent samples 上の GP posterior を積分する必要があります。

`JointVAEGP.uncertainty_aware_posterior()` はこの考え方を Monte Carlo 近似し、
latent-sample posterior mixture の moment を近似します。

通常の deterministic posterior と uncertainty-aware posterior は同じ semantics ではありません。

## 9. Mixed neural representation

categorical code を continuous neural encoder へそのまま入力すると、カテゴリ値の数値距離を
意味のある Euclidean geometry として学習してしまう可能性があります。

robotorchan の Mixed neural model は原則として

```text
continuous X
 -> neural encoder
 -> latent continuous Z

categorical X
 -> passthrough
 -> native categorical covariance
```

と分離します。

対象には frozen model と joint model の両方があります。

例えば、

- `MixedAutoEncoderGP`
- `MixedSupervisedAutoEncoderGP`
- `MixedVAEGP`
- `MixedSupervisedVAEGP`
- `MixedJointEncoderGP`
- `MixedHybridAutoEncoderGP`

などです。

`latent_dim` は continuous input dimension を超えないという制約を持つモデルがあります。

## 10. MultiTask / structural features

MultiTask model では task identity は representation-learning target ではなく structural feature
です。

したがって long-format MultiTask では task column を continuous neural encoder へ暗黙に
混ぜないことが重要です。

同様に Mixed model では categorical feature、MultiFidelity model では fidelity feature の
役割を representation features と区別する必要があります。

named class の全直積を作ること自体が目的ではなく、各 feature role を保持した composition が
重要です。

## 11. Frozen reducer と joint model の比較

| 項目 | Frozen AE/VAE | Joint / Hybrid |
| --- | --- | --- |
| Encoder fitting | GP構築前 | GPと同時 |
| GP objectiveからencoderへの勾配 | なし | あり |
| Latent basis | GP fitting中は固定 | training中に変化 |
| 学習契約 | reducer fit + GP fit | `training_loss()` |
| Reconstruction regularization | reducer側 | Hybrid等でjoint |
| 実装複雑性 | 比較的低い | 高い |

frozen model は安定性と責務分離に優れ、joint model は prediction-aware representation を
直接最適化できる代わりに optimization landscape が複雑になります。

## 12. BoTorch compatibility

frozen neural reducer は `ReducedGP` infrastructure を使うため、public posterior は
original-space candidate `X` を受け、内部で encode します。

joint model も public input は original space を基本とし、`forward` 内で learnable encoder を
通します。

重要なのは、encoder が存在しても acquisition function から見た model interface は
BoTorch posterior contract を維持することです。

一方で joint auxiliary loss は標準 MLL fit helper だけでは表現できないため、
training loop は `training_loss()` contract に従います。

## 13. Bayesian optimization での注意

neural representation は柔軟ですが、BO は一般にデータ数が少ないため、deep representation の
自由度が常に有利とは限りません。

比較の順序としては、

```text
standard GP
 -> PCA / PLS
 -> frozen AE / VAE
 -> supervised reducer
 -> joint / hybrid representation
```

のように complexity を段階的に上げると、追加表現力の価値を評価しやすくなります。

特に確認すべき点は、

- latent dimension sensitivity
- random initialization sensitivity
- reconstruction / supervised / GP loss の balance
- posterior calibration
- encoder drift
- representation uncertainty
- BO trajectory の再現性

です。

## 14. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- `AutoEncoderGP` は frozen deterministic reducer
- `VAEGP` は frozen posterior-mean latent reducer
- Supervised AE/VAE は reducer pretraining で `Y` を利用
- `JointEncoderGP` は GP objective を encoder まで伝播
- `HybridAutoEncoderGP` は GP loss + reconstruction regularization
- joint neural model の共通学習契約は `training_loss()`
- VAE latent uncertainty は deterministic posterior では自動伝播しない
- `uncertainty_aware_posterior()` は latent sampling による近似を提供
- Mixed neural model は continuous encoding と categorical covariance を分離
- structural feature を neural continuous representation と混同しない
- public posterior interface は original input space を基本とする

詳細は [high-dimensional inputs](../models/high_dimensional_inputs.md) を参照してください。

## 参考文献

1. Hinton, G. E. and Salakhutdinov, R. R. (2006).
   Reducing the dimensionality of data with neural networks.
   *Science*, 313(5786), 504-507.
2. Kingma, D. P. and Welling, M. (2014).
   Auto-Encoding Variational Bayes.
   *International Conference on Learning Representations*.
3. Snoek, J., Swersky, K., Zemel, R. S., and Adams, R. P. (2014).
   Input warping for Bayesian optimization of non-stationary functions.
   *International Conference on Machine Learning*.
4. Wilson, A. G., Hu, Z., Salakhutdinov, R., and Xing, E. P. (2016).
   Deep kernel learning.
   *Artificial Intelligence and Statistics*.
5. Damianou, A. and Lawrence, N. D. (2013).
   Deep Gaussian processes.
   *Artificial Intelligence and Statistics*.
6. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)
- [High-dimensional GP](08_high_dimensional_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)
- [Expressive GP](22_expressive_gp.md)

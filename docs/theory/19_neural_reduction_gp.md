# Neural Representation Learning for Gaussian Processes

[次元削減章](18_dimensionality_reduction_gp.md)の合成カーネルと、[GP章](02_gaussian_process.md)の周辺尤度を前提とします。
本章では固定した表現、教師あり事前学習、GPと同時に学習する表現の違いを数式で考えます。

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

Neural reduction は学習方法により大きく2種類に分けられます。

1. encoder を事前学習して固定し、その後 GP を学習する frozen-reducer model
2. GP objective の勾配を encoder まで伝える jointly-trained model

この区別は学習 objective、posterior semantics、更新方法のすべてに影響します。

### 最初に3種類の「Yとの関係」を分ける

neural reductionでは、encoderがいつYやGP objectiveを見るかが重要です。

| family | representation学習時にYを使う | GP objectiveでencoderを更新 |
| --- | --- | --- |
| AE / VAE | いいえ | いいえ |
| Supervised AE / VAE | はい | いいえ |
| Joint Encoder / Hybrid / Joint VAE | はい | はい |

Supervised reducerがYを使うことと、GPとjoint trainingすることは同じではありません。
前者は**GP構築前のrepresentation fitting**、後者は**GP training loopそのもの**です。

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

Autoencoder GP は encoder を reducer として先に学習し、その latent coordinate を固定して
GP を構築します。

```text
観測入力
 -> fit AE once
 -> frozen encoder
 -> latent 観測入力
 -> GP
```

GP marginal likelihoodの勾配は、この事前学習済みencoderの表現学習には戻りません。

つまりfrozen AEが保存しようとするのは主にXのreconstruction informationです。
それが目的関数Yに重要な情報と一致するかは別途validationする必要があります。

この点が joint encoder GP との本質的な違いです。

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

VAE GP は frozen VAE の posterior-mean latent representation を GP input として使います。

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

Supervised Autoencoder GP は reducer の事前学習時に reconstruction loss に加えて
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

Supervised VAE GP は VAE objective に outcome-aware supervision を追加します。

概念的には

```text
L
  = L_reconstruction
  + beta * L_KL
  + lambda_y * L_supervised
```

です。

ここで $\beta\ge0$ は潜在事前分布への正則化、$\lambda_y\ge0$ は応答予測の損失の重みです。

通常の Supervised VAE GP もfrozen reducer modelであり、GP MLLとencoderをjoint training
するモデルではありません。

Supervised AE / VAEではYを使ってreducerをfitするため、cross-validationではencoder fittingも
training fold内へ閉じます。全dataでencoderをfitしてからGPだけをfold分割するとleakageになります。

## 5. joint encoder GP

joint encoder GP は encoder を predictive GP model の一部として扱います。

```text
X
 -> encoder_theta(X)
 -> GP
 -> marginal likelihood
```

GP marginal likelihood の勾配が `theta` まで伝わるため、latent representation は
GP predictive objective によって更新されます。

基本training objectiveは

~~~text
L_joint = - log p(Y | encoder_theta(X))
~~~

です。

例えば零平均GPで、表現のパラメータを $\theta$、観測共分散を
$K_y(\theta)=K_g(e_\theta(X),e_\theta(X))+\sigma^2I$ とします。
カーネルのハイパーパラメータを固定しても、周辺対数尤度の勾配は

$$
\frac{\partial\log p(y\mid\theta)}{\partial\theta_j}
=\frac12\operatorname{tr}\left[
(\alpha\alpha^\top-K_y^{-1})\frac{\partial K_y}{\partial\theta_j}\right],
\qquad \alpha=K_y^{-1}y
$$

となります。これは二次形式と対数行列式を微分した式です。
連鎖律で表現へ勾配が戻るため、単なる再構成とは異なる目的で入力の幾何が変わります。
ただし $\theta$ を点推定する通常の手順では、その重みの事後不確実性まで積分したことにはなりません。
平均関数も $\theta$ に依存する場合は、平均の微分による項も必要です。

## 6. Hybrid Autoencoder GP

GP MLL のみで representation を学習すると、予測に有利でも input structure を強く潰した
latent space になる可能性があります。

Hybrid Autoencoder GP は decoder を追加し、

```text
L_hybrid
  = - log p(Y | encoder_theta(X))
  + lambda_rec * L_reconstruction
```

を使います。

ここで $\lambda_{rec}=0$ なら再構成による正則化はなくなります。
再構成損失を標準化入力で計算するか、物理単位で計算するかによって各座標の重みが変わるため、尺度を明示します。

## 7. Joint VAE

Joint VAE GP 系では probabilistic encoder と GP predictive objective を同時に扱います。

VAE の

- reconstruction
- KL regularization
- stochastic latent representation

と GP likelihood / marginal objective を組み合わせます。

重要なのは、joint VAE が単なる VAE GP の別名ではないことです。

VAE GP は fitted reducer を固定して GP を構築するのに対し、joint VAE では encoder が
predictive training の一部です。

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

標本数をSとし、近似標本ごとのGP予測平均を $\mu_s$、共分散を $\Sigma_s$ とすると、
全分散により混合共分散は

$$
\bar\Sigma=\frac1S\sum_{s=1}^S\Sigma_s+
\frac1S\sum_{s=1}^S(\mu_s-\bar\mu)(\mu_s-\bar\mu)^\top,
\qquad \bar\mu=\frac1S\sum_s\mu_s.
$$

例えば零平均で $\operatorname{Var}(Z)=1$ のコードに対して、関数が $g(z)=z^2$ なら
$g(\mathbb E[Z])=0$ と $\mathbb E[g(Z)]=1$ は異なります。
平均コードへの代入は、非線形な予測を積分する操作ではありません。
さらに、予測時のコードだけを標本化しても、学習入力の表現や表現パラメータの不確実性まで積分できるとは限りません。

## 9. Mixed neural representation

categorical code を continuous neural encoder へそのまま入力すると、カテゴリ値の数値距離を
意味のある Euclidean geometry として学習してしまう可能性があります。

Mixed neural representationでは、カテゴリ値の順序や数値距離を誤って仮定しないよう、原則として

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

- 混合入力 Autoencoder GP
- 混合入力 Supervised Autoencoder GP
- 混合入力 VAE GP
- 混合入力 Supervised VAE GP
- 混合入力 Joint Encoder GP
- 混合入力 Hybrid Autoencoder GP

などです。

潜在次元は表現の容量を決めます。削減を目的とする場合は連続入力の次元以下を選びますが、
次元を増やしても情報量が自動的に増えるわけではありません。具体的な設定制約は実装ガイドで扱います。

### Mixedではreconstruction対象もcontinuous部分

Mixed joint modelではcategoryをencoderへ入れないだけでなく、
混合入力 Hybrid Autoencoder GP のreconstruction lossもcontinuous columnsを対象にします。

~~~text
continuous X
  → encoder
  → decoder
  → reconstruct continuous X

categorical X
  → native categorical covariance
~~~

これによりcategory codeをEuclidean reconstruction targetとして扱うことも避けています。

## 10. MultiTask / structural features

MultiTask model では task identity は representation-learning target ではなく structural feature
です。

したがって long-format MultiTask では task column を continuous neural encoder へ暗黙に
混ぜないことが重要です。

同様に Mixed model では categorical feature、MultiFidelity model では fidelity feature の
役割を representation features と区別する必要があります。

組合せを考える際は、表現学習の対象と構造を表す識別子の役割を保ちます。
タスク共分散と表現上のカーネルを積にする方法や、共通入力格子でKronecker構造を使う方法があります。

long-format MultiTaskではtask featureをencoderから除外し、data featuresだけをlatentへ写像してから
task identityを再結合します。Kronecker形式ではtask identityがXの列ではなくY側のtask axisなので、
観測入力のdata featuresをencoderへ渡せます。

混合マルチタスクでは、カテゴリ設計変数とタスクの識別子を連続表現から分ける構成も考えられます。
これは一つの統計的な設計であり、すべての組合せが同じ仮定や推論方法になるわけではありません。

## 11. Frozen reducer と joint model の比較

| 項目 | Frozen AE/VAE | Joint / Hybrid |
| --- | --- | --- |
| Encoder fitting | GP構築前 | GPと同時 |
| GP objectiveからencoderへの勾配 | なし | あり |
| Latent basis | GP fitting中は固定 | training中に変化 |
| Reconstruction regularization | reducer側 | Hybrid等でjoint |
| 実装複雑性 | 比較的低い | 高い |

frozen modelとjoint modelは学習objectiveが異なります。joint modelはGP predictive objectiveを
representationへ直接伝えられますが、encoderとkernel hyperparameterを同時に同定するため
optimization landscapeも変わります。どちらが良いかはデータ量・representation assumption・
validation結果で判断します。

どちらも元入力 $x$ に対する合成関数 $g(e_\theta(x))$ として定義できます。
表現の学習でGPの入力次元が下がっても、元入力を探索する獲得関数の変数が自動的に減るわけではありません。
実際の処理と設定範囲は[実装ガイド](../guide/19_neural_reduction_gp.md)を参照してください。

## 13. Bayesian optimization での注意

neural representation は柔軟ですが、BO は一般にデータ数が少ないため、deep representation の
自由度が常に有利とは限りません。

model familyは単純な性能順位ではなく、どのrepresentation assumptionを置くかで比較します。

| assumption | 対応する考え方 |
| --- | --- |
| Xの再構成構造をlatentへ残したい | frozen AE |
| latent distributionをregularizeしたい | frozen VAE |
| Y-awareな事前表現を使いたい | supervised AE / VAE |
| GP objectiveから表現を学びたい | Joint Encoder |
| GP objectiveとX再構成を両立したい | Hybrid AE |
| joint trainingにlatent distributionも持たせたい | Joint VAE |

特に確認すべき点は、

- latent dimension sensitivity
- random initialization sensitivity
- reconstruction / supervised / GP loss の balance
- posterior calibration
- encoder drift
- representation uncertainty
- BO trajectory の再現性

です。

## 15. この章で覚えておくこと

表現の固定、教師あり事前学習、GPとの同時学習は異なる学習方法です。
教師ありのAE・VAEが応答を使うことと、GPの周辺尤度で表現を更新することは同じではありません。
同時学習ではGPから表現へ勾配を伝え、Hybrid型では連続入力の再構成も目的に加えます。

VAEの平均コードへの代入と、潜在分布を標本化して積分する予測を区別します。
後者では成分内の分散に加え、予測平均の食い違いも不確実性へ反映されます。
カテゴリやタスクを通常の連続入力へ無条件に混ぜず、自由度の大きさだけで少標本BOの性能を判断しません。

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

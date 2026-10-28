# Expressive Gaussian Process Models

[カーネル章](03_kernel.md)、[変分GP章](09_variational_gp.md)、[ニューラル表現章](19_neural_reduction_gp.md)を前提とします。
本章では、決定論的な表現の学習、確率的な関数合成、事前カーネルの設計を区別します。
入力次元をD、GP階層の層数をLとします。
以下の表のExact GPは、正規観測尤度と固定した表現・ハイパーパラメータの下での条件付き推論を指します。

## 直感

標準 Gaussian Process は kernel によって関数の smoothness や correlation structure を表します。

RBF / Matérn のような標準 kernel は強力ですが、

- learned nonlinear representation
- hierarchical stochastic representation
- neural-network-derived prior
- periodic / quasi-periodic / multi-frequency structure

を明示的に表現したい場合があります。

Expressive GPは、表現力の源が異なる複数のモデル群として整理できます。

### 最初に「何を柔軟にしているか」を分ける

expressive GPでは、複雑さを追加する場所が異なります。

| Family | 柔軟にする場所 | latent representation |
| --- | --- | --- |
| Joint neural GP | kernelへ入る決定論的feature map | deterministic |
| Deep GP | GP mappingそのものを階層化 | stochastic |
| Infinite-width BNN GP | kernel priorの関数形 | explicit latentなし |
| Spectral Mixture GP | stationary kernelのspectral density | explicit latentなし |

したがって「neural」という語が付くモデルでも、有限networkのweightを学習するのか、
GP layerを確率的に積むのか、無限幅極限のanalytic kernelを使うのかで意味が異なります。

## 1. Expressiveness は1種類ではない

この章で扱う主な構造は次です。

| Family | 表現力の源 | 推論 |
| --- | --- | --- |
| Joint neural GP / DKL | learned deterministic features | Exact GP |
| Deep GP | stochastic GP hierarchy | Variational / MC |
| Infinite-width BNN GP | neural-network kernel prior | Exact GP |
| Spectral Mixture GP | spectral density mixture | Exact GP |

これらはすべて「複雑な関数を表せる」という点では共通しますが、posterior semantics と
training contract は大きく異なります。

## 2. Deep Kernel Learning / Joint neural GP

決定論的 neural feature extractor `g_theta` を使うと、

```text
z = g_theta(x)

k_DKL(x, x')
  = k_GP(
      g_theta(x),
      g_theta(x')
    )
```

と考えられます。

```text
x
 -> neural encoder
 -> latent z
 -> Exact GP
```

frozen Autoencoder GP のような二段階 reduction と違い、GP marginal likelihood の勾配が
encoder へ伝わります。

詳細は [Neural Representation GP](19_neural_reduction_gp.md) を参照してください。

## 3. DKL と neural reduction の境界

同じ neural encoder を使っていても、

```text
frozen AE
 -> representation learning
 -> freeze
 -> GP fitting
```

と

```text
joint encoder GP
 -> encoder + GP joint optimization
```

は別の statistical model です。

前者はGPの学習中に固定された座標変換、後者は周辺尤度などの予測目的で更新するカーネルの幾何として理解できます。

このため joint encoder GP は neural-reduction章と expressive-model章の両方に関係しますが、
同じ説明を重複させるのではなく役割を分けます。

## 4. Mixed Joint neural GP

混合入力 Joint Encoder GP は continuous features だけを encoder に渡し、categorical features を
native mixed covariance 側へ保持します。

```text
continuous X
 -> encoder
 -> latent continuous Z

categorical X
 -> native categorical covariance
```

カテゴリコードを連続 neural geometry へ暗黙に混ぜないことが重要です。

## 5. Deep Gaussian Process

Deep GP は GP mapping 自体を階層化します。

```text
x
 -> GP_1
 -> h_1
 -> GP_2
 -> ...
 -> y
```

中間表現 `h_l` が stochastic variable である点が DKL と本質的に異なります。

階層を周辺化すると一般に単純な Gaussian posterior にはならないため、variational inference と
Monte Carlo approximation が必要になります。

## 6. Deep GPの変分表現

Deep GPでは複数の確率的GP写像を合成します。中間層の関数値を積分した予測分布は一般に単一の解析的Gaussian分布にならず、サンプリングにより近似します。

Inducing-point変分分布を各層に導入することで計算を近似できますが、層数・誘導点数・サンプル数が近似精度と計算量に影響します。

## 7. Deep GPの変分推論

Deep GPの周辺尤度は中間層の確率的写像により一般に解析的に計算できません。変分推論では、誘導変数の近似事後分布を導入し、ELBOを最大化します。

$$
\mathcal L_{\mathrm{ELBO}}=\mathbb E_{q(F)}[\log p(Y\mid F)]-\mathrm{KL}(q(U)\|p(U))
$$

ここで $F$ は最終層の潜在値、$U$ は各層の誘導変数をまとめたものです。
層ごとの事前分布と変分分布をそれぞれ積に分解する構成では、KL項は
$\sum_{\ell=1}^L\operatorname{KL}(q(U_\ell)\Vert p(U_\ell))$ となります。
期待対数尤度は中間層の不確実性を通じて計算し、単に各層の平均を次の層へ代入する操作とは異なります。
例えば第1層の値が平均零・分散1でも、第2層の平均関数が二乗なら、期待値は1であって零ではありません。
この期待値はモンテカルロ近似される場合があります。候補間の依存を保つには各層の標本も同時に生成する必要があります。
正規尤度GPの周辺尤度最大化とは、推論対象も近似誤差も異なります。

## 8. MultiTask DeepGP

task identity は単なる continuous scalar input として扱うのではなく、学習可能な task
representation として stochastic hierarchy へ結合します。

```text
data features
 + task representation
 -> DeepGP hierarchy
```

このため standard マルチタスク GP の exact task covariance と同一の構造ではありません。

## 9. Mixed入力を扱うDeep GP

カテゴリ変数を確率的階層に組み込む際には、カテゴリ固有の共分散、適切な符号化、学習可能な埋め込みなどの異なる構造仮定があります。カテゴリ番号を無条件に連続値として扱うと、存在しない順序・距離を導入するため注意が必要です。

## 10. Mixed × MultiTask DeepGP

一つの構成では、連続入力を標準化し、カテゴリとタスクには別々の埋め込みを与えてから確率的な階層へ結合します。
カテゴリの意味とタスク間の転移の仮定を同じものとして扱わない点が重要です。
埋め込みを点推定する場合、その座標の不確実性まで完全に積分したことにはなりません。

## 11. Infinite-width neural network GP

有限幅 neural network の width を無限大へ取ると、適切な weight prior の下で function prior が
Gaussian Process に収束する場合があります。

初期 covariance は概念的に

```text
K^0(x, x')
  = sigma_b^2
    + sigma_w^2
      <x, x'> / D
```

で、各 hidden layer で ReLU covariance map を再帰適用します。

ReLUを $\varphi(u)=\max(u,0)$ とし、独立な零平均正規分布の重みとバイアスで、入力数に応じて重みを正規化した全結合層を考えます。
層 $\ell$ の共分散を $K^\ell$、対角値を $q_x=K^\ell(x,x)>0$ として、
$\vartheta=\arccos(K^\ell(x,x')/\sqrt{q_xq_{x'}})$ と置くと、
同時正規変数のReLU積の期待値から次の再帰を得ます。

$$
K^{\ell+1}(x,x')=\sigma_b^2+
\frac{\sigma_w^2\sqrt{q_xq_{x'}}}{2\pi}
\{\sin\vartheta+(\pi-\vartheta)\cos\vartheta\}.
$$

対角値が零の場合は連続な極限を用います。例えばバイアスなしで同一点なら
$\vartheta=0$ なので、次の層の分散は $\sigma_w^2q_x/2$ です。
重み分散・活性化の規約によって層を重ねたときの分散の増減が変わるため、深さだけで表現力は比較できません。
有限幅の学習済みネットワークや、ニューラル接線カーネルによる学習と同じモデルではありません。

## 12. 無限幅ニューラルネットワーク由来のGP

適切な重み事前分布と幅の極限の下で、ニューラルネットワークの関数事前分布はGPに収束します。得られるNNGP共分散を通常のGPに用いる場合、有限幅ネットワークの重みを直接学習するわけではありません。

深さ、重み分散、バイアス分散は共分散関数の形状を決めます。これらを固定するか推定するかは推論設計の選択であり、無限幅極限そのものの性質とは区別します。

## 13. Infinite-width BNN MultiTask / Kronecker

NNGPカーネルを入力側に、タスク共分散を出力側に置く分離可能な構成を考えられます。
タスクを添字付き観測として扱う場合、その番号をNNGPの通常の連続入力へ混ぜません。
共通入力格子なら潜在共分散は入力因子とタスク因子のKronecker積になりますが、
観測ノイズや欠測を含む行列が同じ積に分解できるとは限りません。

## 14. Mixed Infinite-width BNN

混合入力では、連続入力のNNGP共分散とカテゴリ共分散の役割を分けて、和や積として組み合わせられます。
カーネルがNNGPに由来することは、カテゴリに数値距離を仮定する理由にはなりません。

### NNGP kernelはstationary kernelとは限らない

Infinite-width ReLU kernelはinner productや各入力自身のnormに依存するため、一般に
RBFのようなtranslation-invariant stationary covarianceとは異なります。

したがって「Exact GPだからstationary」という対応はありません。Exact / variationalは
inferenceの分類、stationary / nonstationaryはcovariance structureの分類です。

## 15. Spectral representation

Bochnerの定理では、連続な定常共分散は有限な非負のスペクトル**測度** $S$ により
$k(\tau)=\int e^{2\pi i\omega\tau}\,dS(\omega)$ と表せます。
測度が絶対連続な場合にスペクトル密度を持ちます。実数値カーネルでは対称な測度を用います。

Spectral Mixture kernel は spectral density を Gaussian mixture で近似します。

1次元の概念形は

```text
k(tau)
  = sum_q
      w_q
      exp(
        -2 pi^2 tau^2 v_q
      )
      cos(
        2 pi tau mu_q
      )
```

です。

ここで $\tau=x-x'$ は入力差、$\mu_q$ は周波数、$v_q>0$ は周波数の分散、$w_q\ge0$ は分散への寄与です。
例えば周波数 $\Omega\sim\mathcal N(\mu,v)$ では、特性関数から
$\mathbb E[e^{2\pi i\Omega\tau}]=e^{-2\pi^2v\tau^2}e^{2\pi i\mu\tau}$ です。
周波数 $\mu$ と $-\mu$ の正規密度を等重みで組み合わせると虚部が消え、表示した余弦の形になります。
重みの和は $k(0)=\sum_qw_q$ であり、分散を1に固定しない限り必ずしも1ではありません。

例えば一成分で $w=1,\mu=2,v=0.01$ なら、基本周期は0.5です。
$\tau=0.5$ で共分散は約0.952、$\tau=0.25$ では約 $-0.988$ となります。
負の共分散も妥当であり、半正定値性はカーネル値が常に非負であることを意味しません。
スペクトルの幅 $v$ は周期的な相関が距離とともに減衰する速さを制御します。

## 16. Spectral Mixture Kernelのモデル化

Spectral Mixture Kernelは、定常共分散関数のスペクトル密度をGaussian混合などで近似することで、複数の周波数成分や準周期性を表現します。混合成分を増やすと表現力は増しますが、過適合や推定不安定性にも注意が必要です。

スペクトルパラメータの初期化方法は最適化の出発点を変えるものであり、異なる事後分布族を定義するものではありません。混合成分の置換対称性のため、個々の成分の識別性にも注意が必要です。

## 17. Stationary であること

Spectral Mixture kernel は柔軟ですが、基本的には stationary covariance family です。

複数の frequency を表現できることと、

```text
ell = ell(x)
```

のように location-dependent covariance を持つことは別です。

nonstationarity が必要なら
[Nonstationary Gaussian Process](17_nonstationary_gp.md)
と区別します。

## 18. Spectral Mixture MultiTask / Kronecker

スペクトル混合カーネルを入力側、タスク共分散を出力側に持つ分離可能な構成を考えられます。
タスクの識別子は周波数を持つ物理座標ではありません。共通入力格子ではKronecker構造を利用できます。
出力を平均して周波数初期値を求める方法もありますが、逆位相のタスクが相殺する場合など、
平均が各タスクのスペクトルを代表するとは限りません。初期化は事前仮定や最終的な共分散と区別します。

## 19. Mixed Spectral Mixture

混合入力では、周波数の意味を持つ連続座標にスペクトル構造を、カテゴリには別の共分散を定義します。

したがって categorical code を frequency axis として解釈しません。

## 20. 4 family の比較

| Model family | Representation | Posterior | 主な構造 |
| --- | --- | --- | --- |
| Joint neural GP | deterministic learned feature | Exact GP | nonlinear learned geometry |
| DeepGP | stochastic GP hierarchy | MC approximation | hierarchical nonlinear mapping |
| Infinite-width BNN GP | analytic NNGP kernel | Exact GP | neural prior |
| Spectral Mixture GP | spectral mixture kernel | Exact GP | stationary multi-frequency |

表現力の高さだけで優劣は決まりません。

data size、optimization stability、posterior calibration、acquisition cost、構造仮定との整合性を
比較する必要があります。

## 21. Bayesian optimization との接続

DeepGP は Monte Carlo posterior を提供するため、MC acquisition との組合せが自然です。

元入力 $x$ 上の効用を最適化する場合、獲得関数は内部表現との合成として元入力上に定義できます。

```text
original candidate X
 -> model-internal representation
 -> posterior
 -> acquisition
```

内部表現を学習したことだけでは、実行可能な低次元の探索写像まで定まったことにはなりません。

これは REMBO / ALEBO / BAxUS のような search-space strategy とは別の責務です。

## 22. High-dimensional model との違い

expressive model と high-dimensional model は重なる場合がありますが同義ではありません。

例えば、

- SAAS: sparse relevance prior
- PCA GP: explicit dimension reduction
- joint encoder GP: learned nonlinear kernel geometry
- Infinite Width BNN GP: neural prior without explicit reduction
- Spectral Mixture GP: spectral stationary structure

です。

したがって model selection は「高次元だから expressive model」という単純な対応ではなく、
どの structural assumption が必要かで判断します。

## 24. この章で覚えておくこと

表現力を増やす場所によって、決定論的な特徴写像、確率的なGP階層、NNGP事前、スペクトル混合を区別します。
Deep GPは潜在層の不確実性を積分するため、特徴を点推定するDKLや、有限ネットワークの重みを直接学習する方法と異なります。
無限幅のReLU共分散は一般に非定常でも、通常のGPの条件付き推論に利用できます。

スペクトル混合は定常性を保ちながら周波数構造を柔軟にします。
成分数や深さを増やしても性能改善は保証されず、仮定・推論誤差・較正を比較する必要があります。
カテゴリとタスクは別の構造として扱い、内部表現の柔軟性と探索空間の削減を混同しません。

## 参考文献

1. Wilson, A. G., Hu, Z., Salakhutdinov, R., and Xing, E. P. (2016).
   Deep kernel learning.
   *Artificial Intelligence and Statistics*.
2. Damianou, A. and Lawrence, N. D. (2013).
   Deep Gaussian processes.
   *Artificial Intelligence and Statistics*.
3. Salimbeni, H. and Deisenroth, M. (2017).
   Doubly stochastic variational deep Gaussian processes.
   *Advances in Neural Information Processing Systems*.
4. Neal, R. M. (1996).
   *Bayesian Learning for Neural Networks*. Springer.
5. Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., and Sohl-Dickstein, J.
   (2018).
   Deep neural networks as Gaussian processes.
   *International Conference on Learning Representations*.
6. Wilson, A. G. and Adams, R. P. (2013).
   Gaussian process kernels for pattern discovery and extrapolation.
   *International Conference on Machine Learning*.
7. Rasmussen, C. E. and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.

## 関連章

- [Neural Representation GP](19_neural_reduction_gp.md)
- [Advanced High-dimensional Models](21_advanced_high_dimensional_models.md)
- [Nonstationary GP](17_nonstationary_gp.md)
- [High-dimensional Search](20_high_dimensional_search.md)

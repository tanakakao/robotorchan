# High-dimensional Acquisition Search

## 9.1 問題設定

Bayesian Optimization では surrogate model の学習だけでなく、

    X* = argmax_X alpha(X)

という acquisition optimization 自体も入力次元 d の影響を受けます。

d が大きくなると、

- global candidate search の volume が急増する
- multistart initialization が良い basin を捉えにくくなる
- q-batch では decision dimension が qd まで増える
- local optimizer が局所解へ依存しやすくなる
- population / sampling backend では必要 candidate 数が増えやすい

という問題が生じます。

したがって「高次元 surrogate」と「高次元 acquisition search」は別の問題です。

## 9.2 Model-side と Search-side の次元削減

高次元 BO では次元を下げる場所を区別する必要があります。

### Model-side reduction

PCA、PLS、AE、VAE、joint encoder GP などで surrogate が扱う representation を変えます。

    X_original -> representation Z -> surrogate

### Search-side reduction

surrogate の public input contract は維持しながら、candidate search だけを低次元化します。

    Z_search -> X_original -> acquisition(model, X_original)

両者は surrogate を変えるか探索座標を変えるかという点で異なります。

## 9.3 Effective dimension

高次元問題でも objective が全 d 次元へ等しく依存するとは限りません。

低次元部分空間 A が存在し、

    f(x) approximately g(A^T x)

とみなせる場合、ambient dimension d より小さい effective dimension d_e を利用できます。

random embedding 系手法は、この低 intrinsic/effective dimension 構造を仮定して
低次元座標から元空間を探索します。

この仮定が成立しない場合、固定低次元 embedding は optimum を十分に表現できない可能性があります。

## 9.4 Latent-space reconstruction

学習済みの次元削減写像を使う場合、latent reconstruction により元空間へ戻します。

optimizer は latent coordinate Z を探索し、acquisition evaluation の直前に

    Z -> reconstruct(Z) -> X_original

と戻します。

現在対応する reconstruction は PCA と Gaussian random projection です。

PCA reconstruction は retained PCA subspace への再構成であり、
削除した component を復元するものではありません。

RandomProjection reconstruction は Moore-Penrose pseudoinverse による
minimum-norm reconstruction であり、元の点の一意な復元ではありません。

## 9.5 Latent bounds と feasible set

元空間 box を latent space へ写した領域は一般に単純な axis-aligned box ではありません。

Latent Space 探索 では linear image の coordinate-wise outer bounds を探索 bounds とします。

したがって latent box 内の点を reconstruct すると元空間 bounds を外れる場合があります。
元空間 box への射影は境界を満たしますが、latent 空間での探索幾何を変化させます。

- feasibility_projection_distance
- latent_round_trip_distance

を保存し、reconstruction / projection distortion を診断できます。

## 9.6 REMBO

REMBO は random embedding により低次元 space を探索する代表的手法です。

dense random matrix A を使い、概念的に

    x = p_X(A z)

と元空間へ写します。

p_X は box domain への projection です。

低 effective dimension の function では、適切な embedded dimension を使うことで
ambient dimension より小さい search problem として acquisition optimization を行えます。

Gaussian random embedding と元空間への射影を組み合わせることができます。

embedding は strategy 作成時に固定され、BO iteration ごとには再生成しません。

## 9.7 REMBO の clipping distortion

REMBO の単純な box projection は実装しやすい一方、

    z_1 != z_2
    but
    p_X(A z_1) = p_X(A z_2)

となる領域を作り得ます。

つまり embedded space の異なる点が同じ boundary point へ collapse し、
search geometry を歪める場合があります。

この問題が ALEBO の feasible-polytope approach を導入する動機の1つです。

## 9.8 HeSBO

HeSBO は dense Gaussian embedding の代わりに sparse signed hash embedding を使います。

各 original dimension i は、

- 1つの embedded dimension h(i) に割り当てられる
- sign s_i in {-1, +1} を持つ

という構造です。

embedding matrix の各 original row はちょうど1つの nonzero element を持ちます。

この sparse structure により、ambient dimension が大きい場合でも
projection の計算と保存を軽量化できます。

embedding を固定して元空間の surrogate を評価する構成も考えられます。

## 9.9 ALEBO

ALEBO は linear embedding を使いながら、REMBO の clipping distortion を避けることを狙います。

embedded coordinate z のうち

    lower <= A z <= upper

を満たす領域だけを feasible polytope として探索します。

したがって optimizer は「大きな latent boxを探索して後でclip」するのではなく、
元空間 box に射影可能な embedded region 内で acquisition を最適化します。

## 9.10 ALEBO の geometry

ALEBO では embedding 後の geometry も重要です。

ALEBO では embedded coordinates 上の surrogate に
full Mahalanobis metric を用います。

Cholesky factor U に対して metric を

    M = U^T U

とすると、distance は概念的に

    (z - z')^T M (z - z')

で測られます。

metric initialization、multi-start MAP、
reference diagonal Laplace approximation、metric sampling、moment-matched acquisition model
までを実装範囲としています。

## 9.11 ALEBO の candidate optimization

ALEBO 探索 は feasible embedded polytope 内で acquisition を最適化します。

探索座標と実験対象の元座標を区別し、候補を評価する空間を明示します。

joint q-batch では feasible embedded initial conditions を使います。

sequential q-batch では各 q=1 step に対して feasible initial conditions を再生成します。

## 9.12 BAxUS

固定 embedding dimension は effective dimension が未知の場合に選択が難しくなります。

BAxUS は sparse embedding と trust region を組み合わせ、
探索中に target subspace を拡張します。

概念的には

    small target subspace
      -> local search
      -> insufficient progress
      -> split embedding bins
      -> larger target subspace

という adaptive process です。

これにより最初から ambient dimension 全体を探索せず、
必要に応じて search capacity を増やします。

## 9.13 BAxUS の状態

BAxUSは探索の進行状況と評価予算に応じて埋め込み次元と局所探索領域を適応させます。
探索が停滞した場合に部分空間を拡張し、探索可能な方向を増やします。

ライブラリ固有の状態管理と対応制約は [高次元探索の実装ガイド](../../optimization/high_dimensional_implementation.md) を参照してください。

---

## 9.14 Nested subspace expansion

BAxUS の sparse embedding では original dimensions が target-space bins に割り当てられます。

subspace expansion では、複数 original dimensions を持つ parent bin を child bins に分割します。

subspace expansion では

- 各 original dimension の sign
- 各 original dimension が1 binだけに属する構造
- 既存 target observation の original-space projection

を保持します。

既存 target coordinate は child bins へ複製されるため、
expansion 前の search space を細分化する nested subspace として扱えます。

## 9.15 BAxUS と Trust Region

BAxUS は embedding method であると同時に local trust-region search を使います。

現在の target-space best observation を center とし、

    center_j +/- weight_j * length

で target bounds を構成します。

length は success / failure feedback により適応します。

Trust Region 一般と TuRBO の詳細は次章で扱います。

## 9.16 Lengthscale-weighted target region

search strategy と surrogate を分離する構成では、
strategy 内で別の target-space GP を fit しません。

original-space acquisition model が ARD lengthscale l_i を公開している場合、
target bin B_j に対して

    l_target,j = 1 / sqrt(sum_(i in B_j) 1 / l_i^2)

という induced metric を使います。

その後、weight の geometric mean が1になるよう正規化します。

適切な original-space ARD lengthscale が取得できない場合は isotropic weight に
fallback します。

## 9.17 BAxUS Thompson Sampling

BAx USThompson Sampling 探索 は target trust region 内に有限 candidate pool を作り、
posterior sample に基づいて候補を選びます。

各 candidate は incumbent から sparse perturbation され、
各 target dimension の perturb probability は

    min(20 / target_dim, 1)

です。

どの dimension も変更されなかった candidate には最低1次元の perturbation を強制します。

target-space pool を original space へ射影し、
共通 surrogate model に対する MaxPosteriorSampling で candidate を選択します。

この strategy でも独立した target-space GP は学習しません。

## 9.18 q-batch と高次元 search

高次元 search では q > 1 により optimizer dimension が qd に増えます。
joint acquisition は候補間の相互作用を含むため、独立に選んだ q 点の集合と
joint optimization の結果は一般に一致しません。

random search を用いる場合も、複数の q-batch を生成して各 batch の
joint acquisition value を比較する定式化が考えられます。

## 9.19 Candidate constraints と reduced search space

元空間の制約集合を $\mathcal{F}=\{x:g(x)\geq 0\}$、埋め込み写像を
$P:z\mapsto x$ とすると、潜在空間の可行集合は
$\mathcal{Z}_{\mathcal{F}}=\{z:g(P(z))\geq 0\}$ です。
射影後の制約を無視すると元の制約付き最適化とは異なる問題になります。

ライブラリ固有の状態管理と対応制約は [高次元探索の実装ガイド](../../optimization/high_dimensional_implementation.md) を参照してください。

---

## 9.20 Mixed / categorical space との関係

random linear embedding は通常、Euclidean continuous geometry を前提にします。

unordered categorical variable を数値座標として embedding すると、
category 間に人工的な距離・順序を導入する危険があります。

したがって continuous high-dimensional embedding と mixed/categorical search は
自動的に互換とはみなしません。

mixed search では categorical semantics を保つ optimizer または
categorical-aware representation が必要です。

## 9.21 Reconstruction と embedding の違い

Latent Space 探索 と REMBO / HeSBO / ALEBO / BAxUS は似ていますが、
低次元座標の由来が異なります。

| Strategy | Low-dimensional space | Mapping |
| --- | --- | --- |
| LatentSpace | fitted data reducer | reconstruction |
| REMBO | random dense embedding | fixed projection + clamp |
| HeSBO | sparse signed hash | fixed projection + clamp |
| ALEBO | random linear embedding | feasible polytope |
| BAxUS | adaptive sparse embedding | projection + subspace expansion |

LatentSpace は観測データから得た representation を探索へ再利用します。

random embedding 系は search geometry を構成するために低次元 space を導入します。

## 9.22 Surrogate と Search Strategy の分離

高次元BOでは、予測分布を定義する代理モデルと、候補の探索領域・幾何を
定義する探索法を区別します。両者を独立に比較する際には、他方の条件を統制します。
ALEBOでは計量を考慮したモデル構造も手法の仮定に含まれます。

ライブラリ固有の状態管理と対応制約は [高次元探索の実装ガイド](../../optimization/high_dimensional_implementation.md) を参照してください。

---

## 9.23 Stateful と Stateless

固定埋め込みを用いる方法と、観測結果に応じて埋め込み・探索領域を
適応させる方法を区別します。BAxUSは後者に該当します。

ライブラリ固有の状態管理と対応制約は [高次元探索の実装ガイド](../../optimization/high_dimensional_implementation.md) を参照してください。

---

## 9.24 Numerical stability

高次元の射影・再構成では数値誤差が蓄積し得ます。
特に rank-deficient な写像の逆問題は不安定になり得るため、
擬似逆や正則化の性質を考慮します。

## 9.25 Strategy 選択の考え方

低い有効次元、射影歪み、評価予算、探索計算量などの仮定に応じて
高次元探索法を選択します。いずれの埋め込み法も常に優れるわけではありません。

ライブラリ固有の状態管理と対応制約は [高次元探索の実装ガイド](../../optimization/high_dimensional_implementation.md) を参照してください。

---

## 参考文献

1. Wang, Z., Zoghi, M., Hutter, F., Matheson, D., & de Freitas, N. (2016).
   *Bayesian Optimization in a Billion Dimensions via Random Embeddings*.
   Journal of Artificial Intelligence Research, 55, 361-387.
2. Nayebi, A., Munteanu, A., & Poloczek, M. (2019).
   *A Framework for Bayesian Optimization in Embedded Subspaces*.
   ICML.
3. Letham, B., Calandra, R., Rai, A., & Bakshy, E. (2020).
   *Re-Examining Linear Embeddings for High-Dimensional Bayesian Optimization*.
   NeurIPS.
4. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019).
   *Scalable Global Optimization via Local Bayesian Optimization*.
   NeurIPS.
5. Papenmeier, L., Nardi, L., & Poloczek, M. (2022).
   *Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces*.
   NeurIPS.
6. BoTorch documentation and tutorials for high-dimensional Bayesian optimization,
   TuRBO, and BAxUS.

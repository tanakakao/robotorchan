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

PCA、PLS、AE、VAE、JointEncoderGP などで surrogate が扱う representation を変えます。

    X_original -> representation Z -> surrogate

### Search-side reduction

surrogate の public input contract は維持しながら、candidate search だけを低次元化します。

    Z_search -> X_original -> acquisition(model, X_original)

robotorchan の search strategy は後者を独立した責務として扱います。

## 9.3 Effective dimension

高次元問題でも objective が全 d 次元へ等しく依存するとは限りません。

低次元部分空間 A が存在し、

    f(x) approximately g(A^T x)

とみなせる場合、ambient dimension d より小さい effective dimension d_e を利用できます。

random embedding 系手法は、この低 intrinsic/effective dimension 構造を仮定して
低次元座標から元空間を探索します。

この仮定が成立しない場合、固定低次元 embedding は optimum を十分に表現できない可能性があります。

## 9.4 LatentSpaceStrategy

robotorchan の LatentSpaceStrategy は fitted reducer による latent reconstruction を使います。

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

LatentSpaceStrategy では linear image の coordinate-wise outer bounds を探索 bounds とします。

したがって latent box 内の点を reconstruct すると元空間 bounds を外れる場合があります。
現在の実装は acquisition evaluation 前と返却前に元空間 box へ clamp します。

このため metadata には

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

robotorchan の REMBOStrategy は fixed dense Gaussian embedding を使い、
元空間 box 外への projection は normalized box で clamp します。

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

robotorchan の HeSBOStrategy では embedding は固定され、
surrogate は original space のままです。

## 9.9 ALEBO

ALEBO は linear embedding を使いながら、REMBO の clipping distortion を避けることを狙います。

embedded coordinate z のうち

    lower <= A z <= upper

を満たす領域だけを feasible polytope として探索します。

したがって optimizer は「大きな latent boxを探索して後でclip」するのではなく、
元空間 box に射影可能な embedded region 内で acquisition を最適化します。

## 9.10 ALEBO の geometry

ALEBO では embedding 後の geometry も重要です。

robotorchan の ALEBOGP は embedded coordinates 上で学習し、
full Mahalanobis metric を用います。

Cholesky factor U に対して metric を

    M = U^T U

とすると、distance は概念的に

    (z - z')^T M (z - z')

で測られます。

robotorchan は projection-conditioned metric initialization、multi-start MAP、
reference diagonal Laplace approximation、metric sampling、moment-matched acquisition model
までを実装範囲としています。

これは単なる acquisition optimizer wrapper ではなく、
ALEBO の search geometry と surrogate geometry を対応させる構成です。

## 9.11 ALEBO の candidate optimization

ALEBOStrategy は feasible embedded polytope 内で acquisition を最適化します。

返却 contract は

    SearchResult.candidates -> original/public space
    metadata["embedded_candidates"] -> embedded space

です。

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

## 9.13 BAxUSState

robotorchan の BAxUSState は ambient dimension と残り evaluation budget から

- initial target dimension
- expansion schedule
- split budget
- failure tolerance

を導出します。

target dimension は利用者が任意に固定する旧方式ではなく、
evaluation budget を含む state transition の一部です。

success / failure に応じて trust-region length を更新し、
restart_triggered になった後に expand_subspace() で embedding を拡張します。

## 9.14 Nested subspace expansion

BAxUS の sparse embedding では original dimensions が target-space bins に割り当てられます。

subspace expansion では、複数 original dimensions を持つ parent bin を child bins に分割します。

robotorchan は expansion 時に

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

BoTorch BAxUS tutorial では target-space GP の ARD lengthscale を使って
trust-region shape を調整します。

robotorchan は search strategy と surrogate を分離しているため、
strategy 内で別の target-space GP を fit しません。

original-space acquisition model が ARD lengthscale l_i を公開している場合、
target bin B_j に対して

    l_target,j = 1 / sqrt(sum_(i in B_j) 1 / l_i^2)

という induced metric を使います。

その後、weight の geometric mean が1になるよう正規化します。

適切な original-space ARD lengthscale が取得できない場合は isotropic weight に
fallback します。

これは BoTorch tutorial と完全に同じ surrogate configuration ではなく、
robotorchan の model/search responsibility separation に基づく設計差です。

## 9.17 BAxUS Thompson Sampling

BAxUSThompsonSamplingStrategy は target trust region 内に有限 candidate pool を作り、
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

高次元 search では q > 1 により optimizer dimension がさらに増えます。

robotorchan の search strategy は返却 candidate を [q, d] に統一し、
joint acquisition は q-batch 全体を評価します。

RandomSearchStrategy では num_samples は point 数ではなく
「評価する random q-batch 数」です。

つまり q > 1 では

    [num_samples, q, d]

を生成し、joint acquisition value が最大の batch を返します。

上位 q 個の independent q=1 candidate を並べる実装ではありません。

## 9.19 Candidate constraints と reduced search space

original-space constraint g(X) を latent coordinate z へ移すには、

    g(project(z))

という constraint mapping が必要です。

linear constraint であっても embedding 後の係数や q-batch semantics を
正しく変換しなければなりません。

robotorchan は現在、暗黙の constraint mapping を行いません。

そのため original-space CandidateConstraints は

- LatentSpaceStrategy
- REMBOStrategy
- HeSBOStrategy
- ALEBOStrategy
- BAxUSStrategy

では unmapped constraint として明示的に拒否します。

これは未検証の座標変換を silently 適用しないための correctness boundary です。

## 9.20 Mixed / categorical space との関係

random linear embedding は通常、Euclidean continuous geometry を前提にします。

unordered categorical variable を数値座標として embedding すると、
category 間に人工的な距離・順序を導入する危険があります。

したがって continuous high-dimensional embedding と mixed/categorical search は
自動的に互換とはみなしません。

mixed search では categorical semantics を保つ optimizer または
categorical-aware representation が必要です。

## 9.21 Reconstruction と embedding の違い

LatentSpaceStrategy と REMBO / HeSBO / ALEBO / BAxUS は似ていますが、
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

robotorchan では高次元 search strategy が surrogate を勝手に置き換えないことを原則とします。

REMBO / HeSBO / BAxUS の通常 strategy は original-space surrogate を共有できます。

これにより benchmark では同じ surrogate / acquisition に対して
search strategy だけを変更し、探索性能差を比較できます。

ALEBO は論文上の metric-aware surrogate geometry が手法の一部なので、
ALEBOGP と strategy の対応を明示的に扱います。

## 9.23 Stateful と Stateless

高次元 search strategy は状態の有無でも分けられます。

| Strategy | State |
| --- | --- |
| OriginalSpace | stateless |
| RandomSearch | stateless |
| LatentSpace | fitted reducer を利用 |
| REMBO | fixed embedding |
| HeSBO | fixed embedding |
| ALEBO | fixed embedding / polytope |
| BAxUS | adaptive state |
| BAxUS Thompson Sampling | adaptive state |

BAxUS では objective evaluation 後に values と target_candidates を
update_state() へ明示的に返す必要があります。

search strategy 自体は objective function を評価しません。

## 9.24 dtype / device contract

search strategy が生成する bounds、embedding、latent coordinates、
diagnostic tensors は public bounds の dtype / device を原則として保持します。

GP-based BO では通常 float64 が推奨されます。

GPU を利用する場合も model、acquisition、bounds、strategy 内 tensor の device を
一致させる必要があります。

## 9.25 Strategy 選択の考え方

| 状況 | 主な候補 |
| --- | --- |
| d がまだ直接探索可能 | OriginalSpace |
| 単純で堅牢な基準が必要 | RandomSearch |
| fitted low-dimensional representation がある | LatentSpace |
| low effective dimension を仮定 | REMBO |
| sparse embedding を使いたい | HeSBO |
| clipping distortion を避けたい | ALEBO |
| effective dimension が未知 | BAxUS |
| BAxUS + posterior sampling | BAxUS Thompson Sampling |

高次元だから常に embedding が優れるわけではありません。

effective dimension assumption、projection distortion、evaluation budget、
acquisition optimization cost を合わせて選択する必要があります。

## 9.26 実装上の境界

高次元 BO の性能は

    surrogate expressiveness
    + acquisition function
    + acquisition search strategy

の組合せで決まります。

robotorchan はこれらを分離し、search strategy の比較時に
surrogate 差を探索アルゴリズム差として誤認しない設計を採ります。

モデル側の高次元対応については model theory、
Trust Region / TuRBO は次章で扱います。

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

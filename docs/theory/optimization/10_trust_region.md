# Trust-region Acquisition Optimization

[獲得関数最適化の基礎](01_foundations.md)と[高次元探索](09_high_dimensional.md)を前提とします。
本章では、局所領域の幾何と、その領域を観測に応じて変更する状態遷移を学びます。

## 10.1 目的

Trust-region Bayesian Optimization は、毎 iteration で global domain 全体を探索する代わりに、
現在の incumbent 周辺へ局所探索領域を置き、その領域を観測結果に応じて適応させます。

acquisition function を `alpha(X)`、global bounds を `X_global` とすると、通常の探索は

    X* = argmax_X alpha(X),  X in X_global

ですが、trust-region search では iteration t ごとに

    X* = argmax_X alpha(X),  X in X_TR,t

を解きます。

重要なのは、trust region が surrogate parameter ではなく **candidate search state** だという点です。

## 10.2 TuRBO の基本構造

TuRBO（Trust Region Bayesian Optimization）は、局所BOと適応的な信頼領域を組み合わせます。
中心を選び、その周辺で候補を生成し、観測後に成功・失敗を判定して領域を拡大・縮小します。

ここでは三つの役割を区別します。状態は領域長や成功・失敗回数、比較基準となる最良効用を保持します。
幾何は、その状態と中心から実際の探索領域を定めます。
予測モデルと獲得関数は、その領域内でどこを評価する価値があるかを定めます。
同じモデルでも領域の適応規則を変えられ、同じ領域でも異なる獲得関数を使えます。

## 10.3 Persistent state

状態には、入力次元 $d\ge1$、一回に観測する候補数 $q\ge1$、領域長 $L$ と上下限、
成功・失敗の連続回数と判定回数、最良効用、再始動の要否や回数を含めます。
必要なら、生の観測値の最良値と状態更新用の最良効用も分けます。

連続失敗で縮小するまでの判定回数を、例えば

$$
\tau_{fail}=\left\lceil\max(4/q,d/q)\right\rceil
$$

と定める設計があります。これは一つの経験的規則であり、収束定理から一意に決まる値ではありません。
$d=10,q=4$ なら3回です。次元や一回の観測数は、領域を変更する速さにも影響します。

## 10.4 Success / failure

completed batch の scalar utility の最大値を `y_new`、state best を `y_best` とします。

meaningful improvement の一例として、

    y_new > y_best + eta * max(1, abs(y_best))

を meaningful improvement とみなします。

ここで $\eta>0$ は改善判定の閾値であり、値の選択は問題に依存します。
初回は比較用の有限な最良値を初期化し、その後この規則を適用します。
成功の判定と最良値の更新は別であり、小さな改善でも最良値を更新する設計が可能です。

$\max(1,|y_{best}|)$ を使うため、観測値の単位を変えると同じ判定になるとは限りません。
例えば $y_{best}=0,\eta=0.01,y_{new}=0.005$ は失敗ですが、値を100倍すると改善0.5は閾値0.01を超えます。
標準化や閾値の尺度も状態規則の一部として定めます。

success なら success counter を増やして failure counter を0へ戻し、
failure なら逆の操作を行います。

## 10.5 Trust-region length update

success counter が tolerance に達すると、

    length <- min(2 * length, length_max)

failure counter が tolerance に達すると、

    length <- length / 2

です。

counter は resize 後に0へ戻ります。

縮小後に

    length < length_min

となると再始動が必要な状態に移ります。

閾値を下回った場合は trust region の再初期化を検討します。

## 10.6 Observed value と state utility

ノイズが大きい場合、生の観測の最良値と、領域更新に使う効用を分けることがあります。
観測を $y_i$、状態用のスカラーを $s_i$ とすれば、観測記録は $\max_i y_i$、
成功判定は新しいバッチの $\max_i s_i$ と以前の基準を使う、という設計です。
同じ値を使う場合は $s_i=y_i$ です。

$s_i$ にノイズ除去後の推定値や頑健な効用を使うと、偶然の大きな観測に追随しにくくなる場合があります。
一方、モデルの再学習で推定値も変わるなら、以前の基準と今回の値が同じ尺度と推定規則に基づくかを確認します。
観測の保存と探索状態の更新を分けても、ノイズの影響が自動的に解消されるわけではありません。

## 10.7 Multi-objective outcome

上の成功判定では、候補ごとに一つのスカラー効用を使います。
多目的観測 $y\in\mathbb R^m,m>1$ には、そのままでは大小関係が定まりません。
重み付きスカラー化、超体積への寄与、頑健な効用など、何を改善とみなすかを先に定めます。

多目的の獲得関数と、領域更新に使うスカラーは別の選択です。
スカラー化の重みや参照点を変える場合は、以前の最良値も同じ規則で再評価するなどして比較可能にします。
異なる効用の数値をそのまま比べて成功・失敗としません。

## 10.8 Center update

中心の移動と領域長の更新は関連しますが、別の操作です。
中心は観測済み最良点やモデルの推定最良点から選べますが、その選択を明示する必要があります。
効用だけを更新しても、新しい中心の座標は決まりません。
中心の更新規則は、ノイズ下でどの点をincumbentとみなすかにも依存します。

## 10.9 Batch-size contract

一回にq点の観測を受け取って一度状態を更新するなら、qは状態遷移の単位です。
例えばq=4に合わせた失敗回数の閾値を、1点ずつ4回の更新へ無条件に流用すると、
同じ観測数でも領域が変化する時点が異なります。

部分バッチや非同期更新も設計可能ですが、何点を一度の判定とするか、
未完了の点をどう扱うか、判定回数をどう調整するかを改めて定義します。
固定バッチの規則を採用する場合は、候補生成と状態更新で同じqを維持します。

## 10.10 Trust-region geometry

global bounds を `[l_d, u_d]`、center を `c_d`、dimension weight を `w_d` とすると、
local half width は

    h_d = 0.5 * length * w_d * (u_d - l_d)

です。

したがって

    lower_d = max(c_d - h_d, l_d)
    upper_d = min(c_d + h_d, u_d)

として global domain に clip します。

`length`は絶対座標幅ではなく、各入力範囲に対する相対尺度です。
この式では $u_d>l_d$、正の領域長と重み、箱内の中心を仮定します。
固定座標は別に扱います。境界付近ではclipにより体積が減るため、同じ長さでも実際の領域体積は中心に依存します。

## 10.11 ARD dimension weights

dimension weights は geometric mean が1になるよう正規化されます。

GP が public input space の ARD lengthscale `ell_d` を持つ場合、
ARD lengthscale から anisotropic trust-region geometry を構成できます。

bounds が与えられる場合は input range で lengthscale を正規化してから weight を作ります。

これにより単位や physical range が異なる入力でも、raw lengthscale をそのまま比較することを
避けます。
相対長さ尺度 $r_j=\ell_j/(u_j-l_j)$ から
$w_j=r_j/(\prod_{i=1}^d r_i)^{1/d}$ とすれば $\prod_j w_j=1$ です。
例えば長さ尺度が $(0.2,20)$、入力範囲が $(1,100)$ なら、相対値はどちらも0.2で重みは両方1です。
切り詰め前の領域体積は $L^d\prod_j(u_j-l_j)$ となり、方向ごとの形と全体の尺度を分けられます。

## 10.12 Reduced-space surrogate との境界

PCA、PLS、AE などの reduced-space model が持つ latent lengthscale は、
original/public input dimension の TuRBO geometry ではありません。

したがって

    latent ARD lengthscale
      != public-space TuRBO dimension weight

です。

latent-space の lengthscale を元空間の重みとして使うには、
適切な座標変換や計量の導出が必要です。
例えば潜在写像 $z=T(x)$ のヤコビ行列を $J_T$、潜在ARDの逆二乗行列を $D_z$ とすれば、
局所的に元空間へ引き戻した計量は $M_x=J_T^\top D_zJ_T$ です。
$z=x_1+x_2$、潜在長さ尺度1なら
$M_x=\begin{pmatrix}1&1\\1&1\end{pmatrix}$ で、方向 $(1,-1)$ の距離は零です。
これは元空間の独立な正の対角重みとは異なります。非線形写像では計量も位置に依存します。

## 10.13 Acquisition optimization path

trust-region search は局所 bounds を定め、その内部で acquisition を最適化します。

したがって TuRBO は acquisition optimizer を置き換えるのではなく、

    global bounds
      -> local bounds
      -> selected optimizer

という geometry layer として働きます。

optimizer backend の gradient-based / derivative-free 等の性質は引き続き適用されます。

## 10.14 Thompson-sampling path

TuRBO は acquisition maximization 以外に posterior sampling path も持ちます。

一つの構成では、局所領域内にSobol点を用いた摂動候補集合を生成します。
次元ごとに中心から変更する確率の例は

    p = min(20 / dim, 1)

です。

各 pool candidate は center の一部 dimension だけを変更し、
何も変更されない row には最低1 dimension の perturbation を強制します。

その集合に対して事後関数の同時標本を生成し、重複を避けてq点を選ぶ設計が可能です。
各点の周辺分布から独立に値を引くと、空間的な相関を持つ同一関数の標本にはなりません。

## 10.15 Pending points

asynchronous BO では未観測の評価中候補を pending set として扱います。
局所候補の生成時には pending locations との重複を避けたり、
fantasy-based acquisition へ pending information を組み込んだりできます。
ただし pending points は観測済みのデータではありません。

## 10.16 Candidate constraints

Trust Region を用いる場合、探索領域と candidate constraints の共通部分が
実行可能集合になります。局所領域内に feasible point が存在しない場合は、
領域の拡張や中心の変更が必要になる場合があります。

有限候補集合を点ごとに filter する方法では、各点の実行可能性は判定できますが、
候補間の関係を定める inter-point constraints を一般には保証できません。

## 10.17 Mixed-variable TuRBO

continuous、integer、categorical variable を同じ Euclidean trust region で縮小すると、
離散変数へ人工的な局所距離を導入します。

一つの設計として、continuous dimensions のみを局所化し、
integer / categorical dimensions の許容集合を維持する方法があります。
この場合、離散割当を列挙または探索し、各割当で連続部分を局所最適化します。

## 10.18 MultiFidelity TuRBO

fidelity coordinate と design coordinate は役割が異なります。
一つの設計として design dimensions のみを局所化し、
fidelity dimensions は元の許容範囲を保つ方法があります。

fidelity にも局所領域を設定する方法は考えられますが、
探索費用と精度のトレードオフを含めた別の設計が必要です。

## 10.19 Restart

trust-region length が下限を下回った場合、局所探索領域を再初期化します。
新しい中心は既存観測の良好点、複数の局所領域、あるいは大域的な探索から
選ぶことができます。restart の具体的な規則はアルゴリズムの設計に依存します。

## 10.20 Restart center

再始動先の抽選に同じ乱数種を使えば、その抽選を再現できます。
探索全体の再現性にはモデル学習などの乱数も関係し、再現できることは大域最適性の保証ではありません。

したがって restart は単なる length reset ではなく、

    local state reset + new search center

として扱われます。

## 10.21 Stateful BO loop

典型的には、まず観測から予測モデルを学び、獲得効用と中心を定めます。
次に状態から局所領域を作り、その領域と候補制約の共通部分で候補を選びます。
実際に観測した後で、定義済みのスカラー効用により成功・失敗、最良値、領域長を更新します。
再始動条件に達していれば、新しい中心と局所状態を用意してから次の候補生成へ進みます。

この順序では、未観測候補を観測済みとして状態更新しません。
予測モデルの再学習、中心の選択、領域長の更新を別の操作として扱うことで、
どの情報が各判断に利用されたかを追跡できます。

## 10.22 TuRBO と BAxUS

TuRBO は original/public input space の局所 trust region を適応します。

BAxUS は trust-region adaptation に加えて、低次元 target subspace 自体を段階的に拡張します。

| Method | Local region | Search dimension |
| --- | --- | --- |
| TuRBO | adaptive | original dimension |
| BAxUS | adaptive | adaptive target subspace |

したがって BAxUS は単なる TuRBO backend ではなく、
adaptive embedding と trust-region search を組み合わせた高次元探索法です。

## 10.23 Stateful optimization の整合条件

- trust-region state と surrogate parameter を区別する
- success / failure は一貫した観測 utility で判定する
- q-batch の更新単位を明確にする
- center と length を同じ座標系で定義する
- latent ARD と元空間の ARD を混同しない
- restart 前後の状態遷移を明確にする

## 10.24 関連章

高次元 embedding と BAxUS は
[High-dimensional Acquisition Search](09_high_dimensional.md) を参照してください。

batch / sequential / one-shot optimization は
[Batch and One-shot Optimization](08_batch_one_shot.md) を参照してください。

制約の一般 contract は [Candidate Constraints](05_constraints.md) を参照してください。

より広い高次元 search strategy の整理は
[High-dimensional Search Strategies](../20_high_dimensional_search.md) を参照してください。

## この章で覚えておくこと

信頼領域は候補の探索範囲を適応させる状態と幾何の組合せです。
成功判定の効用、更新する観測の単位、中心と重みの座標系をそろえることで、一貫した探索になります。
局所領域内の改善や再始動だけで、全領域の大域解を得たとはいえません。

## 参考文献

1. Eriksson, D., Pearce, M., Gardner, J., Turner, R. D., & Poloczek, M. (2019).
   *Scalable Global Optimization via Local Bayesian Optimization*.
   Advances in Neural Information Processing Systems 32.
2. Papenmeier, L., Nardi, L., & Poloczek, M. (2022).
   *Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces*.
   Advances in Neural Information Processing Systems 35.
3. Eriksson, D., & Jankowiak, M. (2021).
   *High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces*.
   Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence.
4. BoTorch documentation and tutorials for TuRBO and high-dimensional Bayesian optimization.

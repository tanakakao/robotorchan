# High-dimensional search：モデル座標と候補探索座標を分離する

## 数理上の目的と座標の区別

[理論の対応章](../../theory/optimization/09_high_dimensional.md)を正本とする。高次元の入力空間 `X ∈ R^D`でsurrogateモデルを縮約す
ることと、獲得関数を低次元の探索変数 `Z ∈ R^d`で最大化することは別の操作である。
[高次元探索の全体ガイド](../20_high_dimensional_search.md)ではモデル内部の座標・探索座標・元の実験座標を区別する。

この章では`src/robotorchan/optim/embedding/`および`optim/latent/`のSearchStrategyを対象とする。最終候補は元の`X`で返すが、
acquisitionが評価する座標はstrategyごとに異なる。REMBO / HeSBO等は投影して元空間のacquisitionを評価し、
ALEBOはembedded-spaceの`ALEBOGP`とacquisitionを使う。

## HeSBO：疎なhash embeddingとclipping

`optim/embedding/hesbo.py::HeSBOStrategy`は各元入力軸を1つのembedding軸へ割り当て、ランダムな符号を付けた疎行列を構成する。
元空間への投影は`Z @ embedding.T`を介して行い、正規化座標を`[-1,1]`へclampしたのち公開boundsの中心・半幅へ変換する。

```text
u_raw = Z @ A.T
u = clamp(u_raw, -1, 1)
X = (lower + upper)/2 + (upper - lower)/2 * u
```

実装はclipping距離も返す。**公開boundsを満たしていることは、clippingの影響がゼロという意味ではない**。同じ元空間点に異なる潜在候補が写ると、潜在空間の獲得関数の景観は変わる。
探索側でBoTorch`optimize_acqf`に委譲していても、投影・clipping・seed管理はrobotorchan側の実装である。

## LatentSpaceStrategy：学習済み再構成を通した獲得関数

`optim/latent/strategy.py::LatentSpaceStrategy`は適合済みの`LatentReconstruction`から`latent_bounds`を得る。
内部`_ReconstructedAcquisition`は`reconstruct(Z)`で元座標へ写し、公開boundsでclampしてから**元空間のacquisition**を評価する。
最終候補も元空間へ復元される。

| 層 | 入出力と責務 | 確認事項 |
| --- | --- | --- |
| 元のsurrogate/acquisition | `X_original`を入力として受ける | モデルposteriorの意味は変えない |
| Latent reconstruction | `Z_search → X_original` | 学習済み変換の次元、再構成誤差 |
| BoTorch optimizer | `Z_search`内の獲得関数最大化へDelegated | `latent_bounds`、q、初期点 |
| Candidate return | 復元後の元空間候補 | bounds、dtype/device、必要なら可行性 |

HeSBOもLatentSpaceStrategyも、原空間の`CandidateConstraints`を数学的な写像なしに潜在座標へ渡さない。
`reject_unmapped_candidate_constraints`が元空間制約を検出すれば`NotImplementedError`となる。
clampした候補が元空間の一般非線形制約を満たすとは限らない。

## REMBO・ALEBO・BAxUSの実装上の差

- **REMBO**
  実装：`optim/embedding/rembo.py`
  幾何学的な契約：固定Gaussian embeddingの列を正規化し、`Z @ embedding.T`を正規化元空間`[-1,1]^D`へclamp、公開boundsへ戻す
  検証上の注意：clampによる多対一対応を許す。bounds内でも一般の入力制約が満たされるとは限らない

- **ALEBO**
  実装：`optim/embedding/alebo.py`
  幾何学的な契約：行列の擬似逆`embedding_pinv`から`A @ z <= b`という埋め込みpolytopeを組み立てる
  検証上の注意：polytope外のZは`project`が拒否。一般の元空間candidate制約は別途写像を要する

- **BAxUS**
  実装：`optim/embedding/baxus.py`
  幾何学的な契約：`BAxUSState`が`target_dim`・trust-region長・分割予算を保持し、sparse subspaceを適応的に拡張する
  検証上の注意：一回限りの埋め込みではなく、評価履歴と状態更新が重要

### REMBO：固定ランダム埋め込み

`REMBOStrategy`は指定seedで`[D,d]`のGaussian行列を生成し、列normで正規化する。
`_REMBOAcquisition`が元入力空間へ候補を投影してから**元のacquisition**を呼び出す。モデルの`posterior`を低次元の別GPへ置き換えるわけではない。
投影は元boundsでclampされ、ソースは`clipping_distance`を計算する。複数の潜在点が同じ元空間点へ写る可能性を、保証された探索多様性と取り違えない。

### ALEBO：制約を持つ潜在polytope

`ALEBOStrategy.linear_constraints`は`A = [embedding_pinv; -embedding_pinv]`、`b = 1`の線形制約を生成する。
`is_feasible(Z)`は擬似逆を通した正規化元座標の絶対値が`1 + atol`以下か判定する。`project(Z)`はpolytope外の入力を`ValueError`で拒否する。
実装には数値境界処理用のclampもあるが、**任意のpolytope外候補をclampで可行にする設計ではない**。

### BAxUS：分割予算付きの適応的状態

`BAxUSState`は`dim`と`eval_budget`を必須とし、`new_bins_on_split`を使って初期target dimensionとsplit回数・予算を定める。
`update_baxus_state`は観測値の最大値と`best_value`を比較し、成功・失敗counterに応じて長さを2倍／半分に更新する。TuRBOの状態と類似するが、
BAxUSの改善閾値は実装上`relative_improvement * abs(best_value)`、
`failure_tolerance`はsplit budgetとtarget dimensionから決まるため、**TuRBOと同一のstate-update式ではない**。

`BAxUSStrategy`はtarget-spaceの観測履歴とsparse embeddingを保持する。局所中心は履歴が無ければ原点、観測があればtarget上のbest点である。
ARD由来の元空間lengthscaleが得られる場合はembeddingに集約してtarget側の幾何重みを作る。
元空間`CandidateConstraints`が潜在変数に正しく写像されない場合は、
これらのstrategyは`reject_unmapped_candidate_constraints`で拒否する。

### 実装上の境界

高次元 BO の性能は

surrogate expressiveness
  + acquisition function
  + acquisition search strategy

の組合せで決まります。

比較実験では surrogate 差を探索アルゴリズム差として誤認しないよう、条件を統制する必要があります。

モデル側の高次元対応については model theory、Trust Region / TuRBO は次章で扱います。

### 公開候補と埋め込み候補の追跡

`SearchResult.candidates`は元の実験座標、`metadata["embedded_candidates"]`は埋め込み座標を表す。
metadataの有無と内容はstrategyごとの契約を確認する。`LatentSpaceStrategy`はlinear imageのcoordinate-wise outer
boundsで探索し、`ALEBOStrategy`はfeasible embedded polytope内で最適化する。
ALEBOではsurrogateのMahalanobis geometryとsearch geometryを対応付けるため、単なるoptimizer wrapperとして交換しない。

### BAxUSの状態と候補pool

`BAxUSThompsonSamplingStrategy`はtarget trust region内に有限candidate poolを作り、posterior sampleで候補を選ぶ。
target-spaceの状態、embedding split、予算管理は`BAxUSState`とstrategyが担当する。
BoTorch tutorialにはtarget-space GPのARDを使う構成があるが、この実装は元空間lengthscaleのembeddingへの集約を使う経路も持つため、
同一のsurrogate設定とみなさない。

PCA、PLS、AE、VAE、`JointEncoderGP`はsurrogateの表現を変える。
REMBO / HeSBO / ALEBO / BAxUSと`LatentSpaceStrategy`は候補探索の座標を扱い、低次元座標の由来や公開空間への写像が異なる。

## 座標写像を担当する実装

[embedding](../../../src/robotorchan/optim/embedding/)はrandom embedding・polytope・
BAxUS状態を、[latent](../../../src/robotorchan/optim/latent/)は学習済みreducerに基づく座標変換を担当します。公開候補を元空間へ戻す責務と、
acquisitionを評価する座標は別です。共通のクラス境界と各strategyの差は[高次元探索の主章](../20_high_dimensional_search.md)へ集約しています。
ここでは前節の射影と可行域の計算を対応付けます。

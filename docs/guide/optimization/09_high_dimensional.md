# High-dimensional search：モデル座標と候補探索座標を分離する

## 1. 数理上の目的と座標の区別

[理論の対応章](../../theory/optimization/09_high_dimensional.md)を正本とする。高次元の入力空間 `X ∈ R^D`でsurrogateモデルを縮約することと、獲得関数を低次元の探索変数 `Z ∈ R^d`で最大化することは別の操作である。[高次元探索の全体ガイド](../20_high_dimensional_search.md)ではモデル内部の座標・探索座標・元の実験座標を区別する。

この章では`src/robotorchan/optim/embedding/`および`optim/latent/`のSearchStrategyを対象とする。モデルとacquisitionの公開入力は元の`X`のままで、探索中だけ`Z`を使用する。

## 2. HeSBO：疎なhash embeddingとclipping

`optim/embedding/hesbo.py::HeSBOStrategy`は各元入力軸を1つのembedding軸へ割り当て、ランダムな符号を付けた疎行列を構成する。元空間への投影は`Z @ embedding.T`を介して行い、正規化座標を`[-1,1]`へclampしたのち公開boundsの中心・半幅へ変換する。

```text
u_raw = Z @ A.T
u = clamp(u_raw, -1, 1)
X = (lower + upper)/2 + (upper - lower)/2 * u
```

実装はclipping距離も返す。**公開boundsを満たしていることは、clippingの影響がゼロという意味ではない**。同じ元空間点に異なる潜在候補が写ると、潜在空間の獲得関数の景観は変わる。探索側でBoTorch`optimize_acqf`に委譲していても、投影・clipping・seed管理はrobotorchan側の実装である。

## 3. LatentSpaceStrategy：学習済み再構成を通した獲得関数

`optim/latent/strategy.py::LatentSpaceStrategy`は適合済みの`LatentReconstruction`から`latent_bounds`を得る。内部`_ReconstructedAcquisition`は`reconstruct(Z)`で元座標へ写し、公開boundsでclampしてから**元空間のacquisition**を評価する。最終候補も元空間へ復元される。

| 層 | 入出力と責務 | 確認事項 |
| --- | --- | --- |
| 元のsurrogate/acquisition | `X_original`を入力として受ける | モデルposteriorの意味は変えない |
| Latent reconstruction | `Z_search → X_original` | 学習済み変換の次元、再構成誤差 |
| BoTorch optimizer | `Z_search`内の獲得関数最大化へDelegated | `latent_bounds`、q、初期点 |
| Candidate return | 復元後の元空間候補 | bounds、dtype/device、必要なら可行性 |

HeSBOもLatentSpaceStrategyも、原空間の`CandidateConstraints`を数学的な写像なしに潜在座標へ渡さない。`reject_unmapped_candidate_constraints`が元空間制約を検出すれば`NotImplementedError`となる。clampした候補が元空間の一般非線形制約を満たすとは限らない。

## 4. Strategyごとの検証範囲

旧[Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md)の「HeSBO, latent-space reconstruction and backend constraint semantics」に、`tests/optim/embedding/test_hesbo_strategy.py`と`tests/optim/latent/test_latent_space_strategy.py`のassertion範囲が整理されている。前者はhash行列の1軸1非零・seed再現性・公開bounds、後者はPCA等からの再構成・元のacquisition契約・不正な形状や制約の拒否を対象とする。

これらは特定の入力に対する静的・契約的検証であり、低有効次元仮定の成立やglobal BOより低いregretを保証しない。またREMBO/ALEBO/BAxUS等のembeddingの独立数値検証も、本章のHeSBO/Latent source照合だけからは主張しない。

[候補空間制約](05_constraints.md)、[Trust region](10_trust_region.md)、[検証索引](../../../docs-internal/verification/theory-guide/optimization.md)を併読する。旧Mappingは検収済み区間以外を含むため保全する。


## 5. REMBO・ALEBO・BAxUSの実装上の差

旧[Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md)の「Coordinate-changing high-dimensional strategies」と、対応する実装ソースを照合した。以下の三方式は、**同じ低次元探索であっても可行域の定義と更新方式が異なる**。

| Strategy | 実装 | 幾何学的な契約 | 検証上の注意 |
| --- | --- | --- | --- |
| REMBO | `optim/embedding/rembo.py` | 固定Gaussian embeddingの列を正規化し、`Z @ embedding.T`を正規化元空間`[-1,1]^D`へclamp、公開boundsへ戻す | clampによる多対一対応を許す。bounds内でも一般の入力制約が満たされるとは限らない |
| ALEBO | `optim/embedding/alebo.py` | 行列の擬似逆`embedding_pinv`から`A @ z <= b`という埋め込みpolytopeを組み立てる | polytope外のZは`project`が拒否。一般の元空間candidate制約は別途写像を要する |
| BAxUS | `optim/embedding/baxus.py` | `BAxUSState`が`target_dim`・trust-region長・分割予算を保持し、sparse subspaceを適応的に拡張する | 一回限りの埋め込みではなく、評価履歴と状態更新が重要 |

### REMBO：固定ランダム埋め込み

`REMBOStrategy`は指定seedで`[D,d]`のGaussian行列を生成し、列normで正規化する。`_REMBOAcquisition`が元入力空間へ候補を投影してから**元のacquisition**を呼び出す。モデルの`posterior`を低次元の別GPへ置き換えるわけではない。投影は元boundsでclampされ、ソースは`clipping_distance`を計算する。複数の潜在点が同じ元空間点へ写る可能性を、保証された探索多様性と取り違えない。

### ALEBO：制約を持つ潜在polytope

`ALEBOStrategy.linear_constraints`は`A = [embedding_pinv; -embedding_pinv]`、`b = 1`の線形制約を生成する。`is_feasible(Z)`は擬似逆を通した正規化元座標の絶対値が`1 + atol`以下か判定する。`project(Z)`はpolytope外の入力を`ValueError`で拒否する。実装には数値境界処理用のclampもあるが、**任意のpolytope外候補をclampで可行にする設計ではない**。

### BAxUS：分割予算付きの適応的状態

`BAxUSState`は`dim`と`eval_budget`を必須とし、`new_bins_on_split`を使って初期target dimensionとsplit回数・予算を定める。`update_baxus_state`は観測値の最大値と`best_value`を比較し、成功・失敗counterに応じて長さを2倍／半分に更新する。TuRBOの状態と類似するが、BAxUSの改善閾値は実装上`relative_improvement * abs(best_value)`、`failure_tolerance`はsplit budgetとtarget dimensionから決まるため、**TuRBOと同一のstate-update式ではない**。

`BAxUSStrategy`はtarget-spaceの観測履歴とsparse embeddingを保持する。局所中心は履歴が無ければ原点、観測があればtarget上のbest点である。ARD由来の元空間lengthscaleが得られる場合はembeddingに集約してtarget側の幾何重みを作る。元空間`CandidateConstraints`が潜在変数に正しく写像されない場合は、これらのstrategyは`reject_unmapped_candidate_constraints`で拒否する。

これらの説明は**ソースレベルの契約照合**である。旧Mapping中の`tests/optim/embedding/test_rembo_strategy.py`、`test_alebo_strategy.py`、`test_baxus_strategy.py`等は検証の追跡先だが、ここでは新たに数値ベンチマークを実行していない。ハイパーボリュームやsimple regretの優位性、任意の制約下での可行候補生成はこの照合からは結論しない。

## 6. HeSBOで低次元候補を実験座標に戻す実行例

次は元の入力6次元を2次元の埋め込み空間で探索する。
`HeSBOStrategy.optimize`は元空間のacquisitionを内部で投影経由で評価し、
返り値の`candidates`を6次元の実験座標に復元する。

```python
import torch
from botorch.acquisition.analytic import PosteriorMean
from robotorchan.models import SingleTaskGP
from robotorchan.optim import HeSBOStrategy

train_X = torch.rand(12, 6, dtype=torch.double)
train_Y = -((train_X[:, :2] - 0.4) ** 2).sum(dim=-1, keepdim=True)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = PosteriorMean(model=model)
bounds = torch.stack((torch.zeros(6, dtype=torch.double), torch.ones(6, dtype=torch.double)))
strategy = HeSBOStrategy(bounds, embedding_dim=2, seed=7, num_restarts=2, raw_samples=16)
result = strategy.optimize(acqf)
original = result.candidates
embedded = result.metadata["embedded_candidates"]
assert original.shape == (1, 6)
assert embedded.shape == (1, 2)
assert strategy.embedding.shape == (6, 2)
assert torch.all(original >= bounds[0])
assert torch.all(original <= bounds[1])
torch.testing.assert_close(strategy.project(embedded), original)
```

元空間への射影はclippingを含み得るため、同じ候補に複数の潜在点が
対応する可能性がある。この例は座標とboundsの契約を検証するもので、
低有効次元仮定の成立や最適化性能の優位性は示さない。

## 理論章から分離した実装上の条件

### 9.26 実装上の境界

高次元 BO の性能は

    surrogate expressiveness
    + acquisition function
    + acquisition search strategy

の組合せで決まります。

比較実験では surrogate 差を探索アルゴリズム差として誤認しないよう、
条件を統制する必要があります。

モデル側の高次元対応については model theory、
Trust Region / TuRBO は次章で扱います。

#### 9.5 Latent bounds と feasible set：実装上の補足

このため metadata には

#### 9.11 ALEBO の candidate optimization：実装上の補足

    SearchResult.candidates -> original/public space
    metadata["embedded_candidates"] -> embedded space

#### 9.16 Lengthscale-weighted target region：実装上の補足

BoTorch BAxUS tutorial では target-space GP の ARD lengthscale を使って
trust-region shape を調整します。

#### 9.16 Lengthscale-weighted target region：実装上の補足

これは BoTorch tutorial と完全に同じ surrogate configuration ではなく、
これはモデルの表現と探索空間の設計を区別する観点から理解できます。

## 数値 API と利用上の補足

これは単なる acquisition optimizer wrapper ではなく、
ALEBO の search geometry と surrogate geometry を対応させる構成です。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

PCA、PLS、AE、VAE、JointEncoderGP などで surrogate が扱う representation を変えます。

LatentSpaceStrategy では linear image の coordinate-wise outer bounds を探索 bounds とします。

ALEBOStrategy は feasible embedded polytope 内で acquisition を最適化します。

## 9.13 BAxUSState

BAxUSThompsonSamplingStrategy は target trust region 内に有限 candidate pool を作り、
posterior sample に基づいて候補を選びます。

LatentSpaceStrategy と REMBO / HeSBO / ALEBO / BAxUS は似ていますが、
低次元座標の由来が異なります。

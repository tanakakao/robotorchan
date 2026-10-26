# High-dimensional search：モデル座標と候補探索座標を分離する

## 1. 数理上の目的と座標の区別

[理論の対応章](../../../../theory/optimization/09_high_dimensional.md)を正本とする。高次元の入力空間 `X ∈ R^D`でsurrogateモデルを縮約することと、獲得関数を低次元の探索変数 `Z ∈ R^d`で最大化することは別の操作である。[高次元探索の全体ガイド](../20_high_dimensional_search.md)ではモデル内部の座標・探索座標・元の実験座標を区別する。

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

旧[Optimization Mapping](../../optimization.md)の「HeSBO, latent-space reconstruction and backend constraint semantics」に、`tests/optim/embedding/test_hesbo_strategy.py`と`tests/optim/latent/test_latent_space_strategy.py`のassertion範囲が整理されている。前者はhash行列の1軸1非零・seed再現性・公開bounds、後者はPCA等からの再構成・元のacquisition契約・不正な形状や制約の拒否を対象とする。

これらは特定の入力に対する静的・契約的検証であり、低有効次元仮定の成立やglobal BOより低いregretを保証しない。またREMBO/ALEBO/BAxUS等のembeddingの独立数値検証も、本章のHeSBO/Latent source照合だけからは主張しない。

[候補空間制約](05_constraints.md)、[Trust region](10_trust_region.md)、[検証索引](../verification/optimization.md)を併読する。旧Mappingは検収済み区間以外を含むため保全する。

# High-dimensional Search Strategies：モデル縮約と候補探索を分ける

## 解決したい問題：モデル次元と探索次元は同じではない

高次元BOでは、GPの`d`次元入力を学ぶ困難と、獲得関数`α(X)`を高次元の候補空間で最大化する困難が重なる。PCAなどでモデル入力を縮約しても、候補を元の`D`次元空間から選ぶなら、獲得関数最適化は自動的に低次元にはならない。[理論第20章](../../../theory/20_high_dimensional_search.md)で定義する次の三つの座標を分けることが出発点である。

| 座標 | 使う場所 | 注意 |
| --- | --- | --- |
| `X_original` | 実際のblack-box実験・public posterior入力 | 元のbounds、物理制約が定義される |
| `Z_model` | `PCAGP`等のsurrogate内部 | 学習済みreducerの座標であり探索用とは限らない |
| `Z_search` | REMBO、HeSBO、BAxUSなどのinner optimizer | 元空間への埋め込み写像・可行性を確認する |

モデル側の[Dimensionality Reduction GP](18_dimensionality_reduction_gp.md)と、本章のsearch-side reductionは役割が異なる。

## 理論から実装：共通の候補生成契約

```text
observations (X_original, Y)
         ↓
model fit → posterior / acquisition
                         ↑
Z_search → embedding / feasibility → X_original
                         ↓
              SearchStrategy.optimize
                         ↓
           SearchResult.candidates [q, D]
                         ↓
             実評価 → 更新・再学習
```

`SearchStrategy.optimize(acq_function, *, q=1) -> SearchResult`が高次元探索の共通入口。原空間候補は`[q,D]`で返り、`acquisition_value`はjoint q-batchのscalarまたは`None`を表す。呼出し側が実験評価とモデル再学習を行い、探索strategyが`surrogate.fit()`や`posterior()`の意味を変えるわけではない。

### 探索手法ごとの幾何と責務

| Strategy | 探索座標と実装 | 何を保持するか |
| --- | --- | --- |
| `OriginalSpaceStrategy` | BoTorch `optimize_acqf`へ委譲 | 元空間boundsとBoTorch形式のcandidate制約 |
| `LatentSpaceStrategy` | PCA / random projectionによるlatent candidates | fitted reducer、再構成結果 |
| `REMBOStrategy` | dense Gaussian random embedding | 固定写像とclippingに伴う多対一境界写像 |
| `HeSBOStrategy` | sparse signed hashing embedding | 各元入力次元に1つの非零係数を割り当てる写像 |
| `ALEBOStrategy` | feasible linear polytope | 元空間clippingではなくpolytope内の探索。ALEBO surrogateとの座標契約 |
| `TuRBOStrategy` | 元空間の局所trust region | `TuRBOState`、現在のcenter、評価後のsuccess/failure |
| `BAxUSStrategy` | sparse embeddingを適応的に拡張 | 部分空間とtrust-region state |
| `BAxUSThompsonSamplingStrategy` | BAxUSの局所pool内でThompson sampling | sparse perturbation、pool、samplingの再現性 |

上記は「同じ`optimize_acqf`を呼ぶだけ」という意味ではない。**Implemented**は埋め込み・trust-region・状態更新やcandidate生成、**Delegated**はBoTorch経由の局所最適化・posterior samplingなど。実装ファミリーは`src/robotorchan/optim/embedding/`と`src/robotorchan/optim/`を辿る。詳細な名前・機能の正本は[高次元探索利用ガイド](../../../optimization/high_dimensional_search.md)である。

## REMBO / HeSBO / ALEBO / BAxUSの違い

REMBOは低次元Gaussian行列から元空間へ写した後、元boundsでclampする。境界で異なる潜在点が同一元空間点へ写る可能性があり、最適化景観が歪む。

HeSBOは各元入力軸を一つの埋め込み軸へ符号つきhashする。dense Gaussian写像より疎だが、衝突や有効次元の仮定に注意が必要となる。

ALEBOは元のbox constraintsから誘導されるfeasible polytopeを使い、不自然なclippingを避ける。単純なoriginal-space acquisition評価と異なり、ALEBOGPとのembedded-coordinate契約を揃える必要がある。

BAxUSは当初の低次元部分空間を固定するだけでなく、探索の進行に合わせてtarget dimensionやembeddingを更新する。したがって一回の静的`optimize()`だけでは、その適応性の妥当性は検証できない。

## TuRBO：状態を持つ局所候補探索

`TuRBOStrategy`は`X_original`上でcenterと局所boundsを維持し、観測を受け取ってtrust-regionの長さを更新する。モデルのARD lengthscaleなどを`dimension_weights`として渡せるが、ローカルの幾何計算とモデル固有のlengthscale抽出は分離する。重みは幾何平均が1になるよう正規化され、局所boundsはglobal boundsへclipされる。

```python
import torch
from robotorchan.optim import TuRBOStrategy

bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
center = torch.tensor([0.5, 0.5], dtype=torch.double)
strategy = TuRBOStrategy(bounds, center=center)
print(strategy.state.length)
```

構築例は実行済みのテスト結果ではない。実際のclosed-loopでは`strategy.optimize(acquisition)`が返した`result.candidates`を**実評価**し、その結果を`strategy.update_state(new_Y, candidates=result.candidates)`へ渡す。batch qの場合は各完了batch単位でstate更新を1回行い、pending pointsのみでは完了観測として更新しない。[TuRBO利用ガイド](../../../optimization/turbo.md)でThompson、async、restartと制約の具体的な契約を確認する。

## 制約と検証の境界

`CandidateConstraints`は評価前から既知の入力制約であり、posterior上の未知outcome constraintsとは区別する。元空間で定義したlinear/nonlinear constraintsを、embedding strategyの潜在座標へそのまま転送すると**違う制約**になる。現行の`LatentSpaceStrategy`、REMBO、HeSBO、ALEBO、BAxUSは元空間candidate constraintsの未写像ケースを明示的に拒否する。対して`OriginalSpaceStrategy`ではBoTorchのintra-point/inter-point制約契約を維持する。Mixed strategyのinter-point nonlinear制約はupstream非対応である。

検証では (1) coordinate mappingとbounds、(2) q-batch shape、(3) candidate feasibility、(4) embeddingの固定／更新、(5) TuRBO state transitionとrestart、(6) seed再現性、(7) acquisition scoreと実際の評価結果を別々に扱う。単なる候補shape、CI成功、Notebookの実行はglobal convergenceやregret保証を与えない。

関連：[理論第20章](../../../theory/20_high_dimensional_search.md)、[探索の最適化ガイド](optimization/09_high_dimensional.md)、[Trust region](optimization/10_trust_region.md)、[Optimization Mapping](../optimization.md)、[Verification Mapping](../verification.md)。既存Mappingはmigration ledgerの検証完了まで保全する。

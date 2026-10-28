# High-dimensional Search Strategies：モデル縮約と候補探索を分ける

## 解決したい問題：モデル次元と探索次元は同じではない

高次元BOでは、GPの`d`次元入力を学ぶ困難と、獲得関数`α(X)`を高次元の候補空間で最大化する困難が重なる。PCAなどでモデル入力を縮約しても、候補を元の`D`次元空間から選ぶなら、獲得関数最適化は自動的に低次元にはならない。[理論第20章](../theory/20_high_dimensional_search.md)で定義する次の三つの座標を分けることが出発点である。

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

上記は「同じ`optimize_acqf`を呼ぶだけ」という意味ではない。**Implemented**は埋め込み・trust-region・状態更新やcandidate生成、**Delegated**はBoTorch経由の局所最適化・posterior samplingなど。実装ファミリーは`src/robotorchan/optim/embedding/`と`src/robotorchan/optim/`を辿る。詳細な名前・機能の正本は[高次元探索利用ガイド](../optimization/high_dimensional_search.md)である。

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

構築例は実行済みのテスト結果ではない。実際のclosed-loopでは`strategy.optimize(acquisition)`が返した`result.candidates`を**実評価**し、その結果を`strategy.update_state(new_Y, candidates=result.candidates)`へ渡す。batch qの場合は各完了batch単位でstate更新を1回行い、pending pointsのみでは完了観測として更新しない。[TuRBO利用ガイド](../optimization/turbo.md)でThompson、async、restartと制約の具体的な契約を確認する。

## 制約と検証の境界

`CandidateConstraints`は評価前から既知の入力制約であり、posterior上の未知outcome constraintsとは区別する。元空間で定義したlinear/nonlinear constraintsを、embedding strategyの潜在座標へそのまま転送すると**違う制約**になる。現行の`LatentSpaceStrategy`、REMBO、HeSBO、ALEBO、BAxUSは元空間candidate constraintsの未写像ケースを明示的に拒否する。対して`OriginalSpaceStrategy`ではBoTorchのintra-point/inter-point制約契約を維持する。Mixed strategyのinter-point nonlinear制約はupstream非対応である。

検証では (1) coordinate mappingとbounds、(2) q-batch shape、(3) candidate feasibility、(4) embeddingの固定／更新、(5) TuRBO state transitionとrestart、(6) seed再現性、(7) acquisition scoreと実際の評価結果を別々に扱う。単なる候補shape、CI成功、Notebookの実行はglobal convergenceやregret保証を与えない。

関連：[理論第20章](../theory/20_high_dimensional_search.md)、[探索の最適化ガイド](optimization/09_high_dimensional.md)、[Trust region](optimization/10_trust_region.md)、[Optimization Mapping](../../.github/project-records/theory-guide/mapping/optimization.md)、[Verification Mapping](../../.github/project-records/theory-guide/mapping/verification.md)。既存Mappingはmigration ledgerの検証完了まで保全する。

## 理論章から分離した実装上の条件

### 21. 実装との対応

現在の理論章で保持すべき実装契約は次です。

- REMBO / HeSBO は search embedding であり fitted input reducer ではない
- embedding strategy は original-space candidate を返す
- ALEBO は target-space feasible polytope を明示的に扱う
- `ALEBOStrategy` と `ALEBOGP` は別責務
- TuRBO は success / failure により trust-region length を適応
- TuRBO は restart state と restart center generation を持つ
- TuRBO は public-space ARD dimension weights を扱う
- reduced-model latent lengthscale を public-space TuRBO weight として誤用しない
- TuRBO は acquisition optimization と Thompson sampling の両経路を持つ
- Mixed TuRBO は discrete dimensions を continuous TR から分離
- MultiFidelity TuRBO は fidelity dimensions を design TR から分離
- constraints / pending pointsはpublic-space semanticsを維持
- REMBO / HeSBO / BAxUS / latent searchはunmapped public constraintsを拒否
- ALEBO internal polytopeはuser candidate constraintではない
- BAxUSはtarget dimensionをstatefully expansion
- BAxUS strategyは別のtarget-space GPを内部でfitせず、利用可能ならoriginal-space ARDからweightを誘導
- BAxUS は acquisition optimization と Thompson-sampling candidate generation を持つ
- state update は objective evaluation / surrogate fitting と分離
- surrogate latent space と search target space は別座標系

詳細は
[high-dimensional search strategies](../optimization/high_dimensional_search.md)
を参照してください。

#### 直感：実装上の補足

robotorchan はこのため、**surrogate-side reduction** と **search strategy** を分離します。

#### 1. Original-space search：実装上の補足

通常の BoTorch optimization は

#### 2. Search embedding：実装上の補足

ただし「acquisitionがどの座標を受け取るか」はstrategyごとに異なります。
REMBO / HeSBO / BAxUSはtarget candidateをoriginal spaceへ写してからoriginal-space acquisitionを
評価します。一方、現行ALEBO workflowはembedded coordinates上の `ALEBOGP` /
acquisitionを使います。

#### 3. REMBO：実装上の補足

robotorchan の `REMBOStrategy` は、

#### 4. HeSBO：実装上の補足

robotorchan の `HeSBOStrategy` も public contract は REMBO と同様で、

#### 6. ALEBO surrogate geometry：実装上の補足

robotorchan では search strategy に加えて `ALEBOGP` があります。

#### 7. TuRBO の基本：実装上の補足


robotorchan の `TuRBOState` は少なくとも

#### 8. TuRBO state update：実装上の補足

robotorchan は observed values と、任意の `state_values` を分けられます。
これにより raw observation とは別の denoised / robust utility で trust-region state を
更新する構成も可能です。

#### 10. ARD trust-region geometry：実装上の補足

robotorchan は dimension weights を geometric mean 1 に正規化し、

#### 11. TuRBO Thompson sampling：実装上の補足

robotorchan は local acquisition optimization に加えて Thompson-sampling candidate generation を
提供します。

#### 14. Constraints と pending points：実装上の補足

現行contractは次のように分かれます。

#### robotorchan BAxUSとtutorial構成の責務差：実装上の補足

#### robotorchan BAxUSとtutorial構成の責務差

#### robotorchan BAxUSとtutorial構成の責務差：実装上の補足

現行robotorchanでは、探索strategy間でsurrogateを共通化できるよう、
BAxUS strategy内部で別のtarget-space GPを毎反復fitしません。

#### robotorchan BAxUSとtutorial構成の責務差：実装上の補足

したがって、BAxUSのsubspace / trust-region adaptationは保持しつつ、
surrogate構成までBoTorch tutorialと完全に同一だと解釈しないことが重要です。

#### 17. BAxUS acquisition optimization と TS：実装上の補足

加えて robotorchan には Thompson-sampling candidate generation の経路があり、
target-space candidate pool を original space へ投影した上で posterior sampling を行います。

#### 18. Stateful strategy contract：実装上の表現

```text
fit surrogate
 -> build acquisition
 -> strategy.optimize(...)
 -> evaluate objective
 -> strategy.update_state(...)
 -> append data
 -> refit surrogate
```

## 数値 API と利用上の補足

- random embedding の保持
- target-space acquisition wrapper
- original-space projection
- target-space optimization
- original-space candidate の返却

default perturbation probability は概念的に

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

したがって `PCAGP` のような data-driven input reducer とは異なります。

`ALEBOStrategy` のpolytope constraintsは**public process constraintではなく**、
projection後のcandidateをbox内へ保つためのinternal constraintsです。

`ALEBOStrategy` は、

`ALEBOGP` は embedded coordinates 上で Euclidean RBF をそのまま使うのではなく、
Mahalanobis geometry を持つ covariance を提供します。

```text
ALEBOStrategy = candidate search geometry
ALEBOGP       = surrogate geometry
```


`TuRBOStrategy.thompson_sample()` はこの local candidate pool と posterior sampling を組み合わせます。

`TuRBOStrategy.optimize_mixed()` は mixed-variable optimization backend と組み合わせて candidate を
生成します。

`TuRBOStrategy.optimize_multifidelity()` はこの geometry を MultiFidelity candidate generation と
接続します。

asynchronous BOではunresolved candidateを `X_pending` として扱う経路があります。
TuRBOではpending pointsはpublic input coordinatesとしてshape / finite validationされます。
candidate feasibilityとpending semanticsは別contractとして確認します。

`BAxUSState` は target dimension と trust-region state を組み合わせます。

`BAxUSStrategy` は embedded target space で acquisition を最適化し、original-space candidate を
返します。

## 専用処理の補足

`TuRBO 探索.thompson_sample()` はこの local candidate pool と posterior sampling を組み合わせます。

`TuRBO 探索.optimize_mixed()` は mixed-variable optimization backend と組み合わせて candidate を
生成します。

`TuRBO 探索.optimize_multifidelity()` はこの geometry を MultiFidelity candidate generation と
接続します。

## 探索手順の実装上の対応

## 12. Mixed TuRBO

Mixed search space では continuous dimension と discrete dimension に同じ trust-region geometry を
適用しません。

`turbo_mixed_trust_region_bounds` は

- continuous design dimensions: local trust region へ縮小
- integer / categorical dimensions: global range を保持

します。

`TuRBOStrategy.optimize_mixed()` は mixed-variable optimization backend と組み合わせて candidate を
生成します。

これによりカテゴリコードを連続 ARD geometry として縮小することを避けます。

## 探索手順の実装上の対応

## 13. MultiFidelity TuRBO

MultiFidelity search でも fidelity coordinate と design coordinate は役割が異なります。

`turbo_multifidelity_trust_region_bounds` は design dimensions を localize しつつ、
fidelity dimensions は global range に保持します。

したがって trust region は

```text
design space    -> local
fidelity space  -> global
```

という構造を取ります。

`TuRBOStrategy.optimize_multifidelity()` はこの geometry を MultiFidelity candidate generation と
接続します。

## 探索手順の実装上の対応

## 14. Constraints と pending points

candidate constraintsはpublic/original input coordinatesで定義されます。そのためsearch coordinateが
変わるstrategyへ同じ係数をそのまま渡すことはできません。

現行contractは次のように分かれます。

| Strategy | public candidate constraints |
| --- | --- |
| TuRBO continuous / MultiFidelity | optimizer経由で対応 |
| TuRBO Mixed | mixed backendの対応範囲で扱う |
| REMBO / HeSBO / BAxUS | unmappedのため明示的に拒否 |
| ALEBO | user constraintは拒否。internal polytopeのみ |
| learned LatentSpace | unmappedのため明示的に拒否 |

つまり「project後にfeasibleか確認すればよい」という一般実装ではありません。
constraintをsearch coordinatesへ厳密に写せないstrategyはsilent approximationを避けて拒否します。

asynchronous BOではunresolved candidateを `X_pending` として扱う経路があります。
TuRBOではpending pointsはpublic input coordinatesとしてshape / finite validationされます。
candidate feasibilityとpending semanticsは別contractとして確認します。

## 探索手順の実装上の対応

## 18. Stateful strategy contract

TuRBO / BAxUS の state は surrogate model の parameter ではありません。

典型的な BO loop は

```text
fit surrogate
 -> build acquisition
 -> strategy.optimize(...)
 -> evaluate objective
 -> strategy.update_state(...)
 -> append data
 -> refit surrogate
```

です。

surrogate model は objective evaluation を所有せず、search strategy も surrogate training を
暗黙に所有しません。

この責務分離により、同じ surrogate / acquisition と異なる search strategy を比較できます。

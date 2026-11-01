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

対応する公開実装の契約は次のとおりです。

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

### 探索空間とacquisitionの入力

通常のoriginal-space searchは、元入力のbounds内でacquisitionを評価する。
`REMBOStrategy`と`HeSBOStrategy`はrandom embeddingを保持し、target-spaceで探索し、
元空間へ投影してoriginal-space acquisitionを評価する。
これはデータからreducerを学習する`PCAGP`とは異なる。
どちらも最終的な`SearchResult.candidates`は元空間へ戻す。

ALEBOでは、embedded coordinates上の`ALEBOGP`とacquisitionを使う。
`ALEBOStrategy`はfeasible polytope内で候補を探し、公開された元空間候補を返す。
このinternal polytopeは、投影後の候補をbox内へ保つための制約であり、
ユーザーのprocess constraintとは別である。

```text
ALEBOStrategy → candidate search geometry
ALEBOGP → Mahalanobis metricによるsurrogate geometry
```

### TuRBOの観測・局所領域・候補生成

`TuRBOState`は長さ、成功・失敗counter、best値、restart情報を保持する。
`state_values`を使う場合は生のobserved valuesと区別し、
denoised / robust utilityを領域更新に使うこともできる。
`dimension_weights`は幾何平均1に正規化され、局所boundsはglobal boundsへclipされる。
元空間の重みとして、reduced modelのlatent lengthscaleをそのまま流用しない。

`TuRBOStrategy.optimize()`は局所acquisition最適化、
`TuRBOStrategy.thompson_sample()`はlocal poolとposterior samplingを使う。
Thompson poolの既定perturbation probabilityは`min(20 / d, 1)`で、
全次元が未変更なら少なくとも1次元を変更する。
状態更新とrestartの詳細は[TuRBO Guide](optimization/10_trust_region.md)を参照する。

### MixedとMultiFidelityの局所領域

| 経路 | 局所化する次元 | global rangeに保つ次元 |
| --- | --- | --- |
| `turbo_mixed_trust_region_bounds` | continuous design | integer / categorical |
| `turbo_multifidelity_trust_region_bounds` | design | fidelity |

`TuRBOStrategy.optimize_mixed()`と`TuRBOStrategy.optimize_multifidelity()`は
このgeometryを対応する候補生成へ接続する。
カテゴリ番号やfidelityを通常の連続design座標と同じように縮小しない。

### 候補制約とpending

| Strategy | 公開空間のcandidate constraints |
| --- | --- |
| TuRBO continuous / MultiFidelity | optimizerの対応範囲で処理 |
| TuRBO Mixed | mixed backendの対応範囲で処理 |
| REMBO / HeSBO / BAxUS | 未写像の制約は明示的に拒否 |
| ALEBO | ユーザー制約は拒否。internal polytopeのみ |
| learned `LatentSpaceStrategy` | 未写像の制約は明示的に拒否 |

投影後に検査するだけで任意の制約を扱えるわけではない。
search coordinatesへ写像していない制約を暗黙の近似に置き換えない。
非同期BOの`X_pending`は未完了の公開座標で、shape・finite値の検証と
candidate feasibilityは別の契約である。

### BAxUSと状態を持つBO loop

`BAxUSState`はtarget dimensionとtrust-region stateを組み合わせる。
`BAxUSStrategy`はtarget-spaceでacquisitionを最適化して元空間へ候補を返す。
Thompson sampling経路では、target-space poolを元空間へ投影してposterior sampleを評価する。

このstrategyは内部で別のtarget-space GPを毎回学習しない。
利用可能ならoriginal-spaceのARDからtarget geometryの重みを誘導する。
subspace / trust-region adaptationを保持していても、
BoTorch tutorialのsurrogate設定と完全に同一ではない。

```text
surrogateを学習
  → acquisitionを構築
  → strategy.optimize(...) またはThompson sampling
  → 実験を評価
  → strategy.update_state(...)
  → 観測を追加してsurrogateを再学習
```

stateはsurrogate parameterではない。実験評価・観測追加・モデルの再学習は呼び出し側が担当し、
strategyが暗黙に所有しない。この分離を保ち、同じsurrogate / acquisitionで
探索法の違いを比較する。

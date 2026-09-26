# 高次元ベイズ最適化の探索戦略

robotorchan では surrogate model と acquisition function と search strategy を分離する。
高次元モデルを変更せず、獲得関数の最適化方法だけを切り替えられる。

## 共通 API

すべての search strategy は `SearchStrategy.optimize(acq_function, *, q=1) -> SearchResult` を共通入口とする。`SearchResult.candidates` は public/original input space の shape `[q, d]` 候補で、`acquisition_value` はその joint q-batch に対応する scalar tensor または `None` である。

search strategy は surrogate の posterior API や reducer lifecycle を変更しない。探索時間は `SearchResult` に含めず、benchmark / 呼び出し側で外部計測する。旧 API 名、deprecated wrapper、互換 alias は提供しない。

## Candidate-space constraint contract

`CandidateConstraints` は探索候補 `X` 自体に課す制約を表す。制約付き獲得関数が扱う
`g(x) <= 0` のような probabilistic output constraint とは別の責務である。

線形制約は BoTorch の `(indices, coefficients, rhs)` 表現をそのまま採用する。
不等式は `sum(X[indices] * coefficients) >= rhs`、等式は
`sum(X[indices] * coefficients) == rhs` と解釈する。独自の行列表現へ変換しないため、
BoTorch が区別する intra-point / inter-point q-batch constraint を保持できる。

Phase 2 では共通 contract のみを導入し、全 strategy が対応済みとはみなさない。
`OriginalSpaceStrategy` は `CandidateConstraints` の線形不等式・線形等式を
`optimize_acqf` へそのまま forwarding する。したがって original/public input space
上の線形制約は BoTorch と同じ意味で利用できる。
1 次元 `indices` の intra-point constraint は q-batch の各候補へ適用され、2 次元
`indices` の inter-point constraint は候補間の関係を表現できる。`OriginalSpaceStrategy`
はこの表現を変換せず保持するため、`q > 1` でも BoTorch の線形 q-batch constraint
semantics をそのまま利用できる。
非線形 candidate constraint は初期値生成や q-batch semantics が異なるため、
線形制約と同時に曖昧な API を公開せず後続 Phase で個別に扱う。

## 戦略

| Strategy | 探索空間 | 状態 | 主な用途 |
| --- | --- | --- | --- |
| `OriginalSpaceStrategy` | 元の d 次元 | なし | 標準 `optimize_acqf`、低〜中次元の基準 |
| `RandomSearchStrategy` | 元の d 次元 | なし | 高次元での単純・堅牢なベースライン |
| `LatentSpaceStrategy` | PCA / RandomProjection 潜在空間 | reducer に依存 | 既知の低次元表現を使う探索 |
| `REMBOStrategy` | dense Gaussian random embedding | 固定 | intrinsic dimension が低いと期待できる場合 |
| `ALEBOStrategy` | feasible linear polytope | 固定 | clipping を避け、線形埋め込みの幾何を保った探索 |
| `HeSBOStrategy` | sparse signed hash embedding | 固定 | 高次元で疎な埋め込みにより acquisition search を軽量化したい場合 |
| `TuRBOStrategy` | 元空間の局所 trust region | あり | 高次元で局所探索を適応的に集中 |
| `BAxUSStrategy` | sparse signed embedding | あり | intrinsic dimension が不明で、探索中に部分空間を拡張したい場合 |
| `BAxUSThompsonSamplingStrategy` | sparse signed embedding | あり | BAxUS の trust region 内で sparse perturbation + Thompson sampling を使う場合 |

### RandomSearch の q-batch

`RandomSearchStrategy` の `num_samples` は候補点数ではなく、評価するランダムな q-batch 数を表す。
`q=1` では `num_samples` 個の点を比較し、`q>1` では shape `[num_samples, q, d]` の候補 batch を生成して、獲得関数が返す joint batch value が最大の 1 batch を返す。
したがって `q>1` でも各点を q=1 として独立評価して上位 q 点を並べる処理は行わない。

### HeSBO

`HeSBOStrategy` は各 original input dimension をランダムに 1 個の embedded dimension へ hash し、係数として `+1` または `-1` を割り当てる固定 sparse embedding を使う。各元次元は embedding 行列のちょうど 1 要素だけが非ゼロになる。REMBO の dense Gaussian embedding と異なり、射影計算が疎で、ambient dimension が大きい場合の探索用 embedding として扱いやすい。

```python
strategy = HeSBOStrategy(
    bounds,
    embedding_dim=5,
    seed=0,
)
result = strategy.optimize(acq_function)
```

surrogate model は original space のままで、embedded coordinate は acquisition 評価時に original space へ射影される。元空間の box を超える射影は normalized `[-1, 1]` box で clamp し、`SearchResult.metadata` に embedded candidate、embedding 行列、clipping distance を保存する。embedding は strategy の生成時に固定され、逐次 BO の途中で学習し直さない。

## ALEBO の使い方

ALEBO は固定線形 embedding を使うが、REMBO のように元空間の box を越えた点を clamp しない。埋め込み座標 `z` から元空間へ線形射影した点が bounds 内に残る領域を polytope として構成し、その領域内で acquisition function を最適化する。

```python
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from robotorchan.models import ALEBOGP
from robotorchan.optim import ALEBOStrategy

strategy = ALEBOStrategy(bounds, embedding_dim=5, seed=0)
train_Z = ...  # strategy の feasible embedded coordinates
train_X = strategy.project(train_Z)
train_Y = objective(train_X)
train_Yvar = torch.full_like(train_Y, 1e-6)

model = ALEBOGP(
    train_Z,
    train_Y,
    train_Yvar,
    projection=strategy.embedding,
)
acquisition_model = model.fit_acquisition_model(
    n_metric_samples=16,
    restarts=10,
    generator=torch.Generator().manual_seed(1),
)
acq = LogExpectedImprovement(model=acquisition_model, best_f=train_Y.max())
result = strategy.optimize(acq, q=1)
```

`ALEBOGP` は embedded coordinates `train_Z` 上で学習し、上三角 Cholesky factor `U` による full Mahalanobis metric `U.T @ U` を使う。観測分散 `train_Yvar` は fixed-noise ALEBO の必須入力である。`projection=strategy.embedding` を渡すと、ALEBO の projection に条件づけた metric 初期化を使う。

`fit_acquisition_model()` は multi-start MAP fitting、reference diagonal Laplace covariance、固定 metric sample、full predictive covariance の Gaussian moment matching を順に行い、BoTorch acquisition function に渡せる model を返す。個別の数値処理を検証・研究したい場合には `metric_hessian_diagonal()`、`metric_laplace_covariance()`、`sample_metric_parameters()`、`marginal_metric_posterior()` も利用できる。

acquisition function は embedded space で評価され、`ALEBOStrategy.optimize()` が feasible polytope 内で最適化した後にだけ original/public input space へ戻す。`SearchResult.candidates` は original space、`metadata["embedded_candidates"]` は embedded space の候補である。q-batch acquisition も同じ contract で扱う。

### ALEBO 実装ステータス

ALEBO の実装範囲は、固定 hypersphere embedding、feasible polytope、projection-conditioned Mahalanobis GP、multi-start MAP、reference diagonal Laplace approximation、metric-marginal posterior、BoTorch acquisition model、q-batch constrained acquisition optimizationまでとする。これらを ALEBO の完成基準とし、追加の full-Hessian 近似や別 sampler などは将来の独立 enhancement として扱う。

### REMBO / ALEBO / BAxUS の使い分け

- REMBO: 最も単純な固定 dense random embedding。box 外への射影は clamp する。
- ALEBO: 固定線形 embedding + feasible polytope + Mahalanobis geometry。clipping による非線形歪みを避けたい場合に使う。
- BAxUS: sparse embedding を探索中に拡張する。intrinsic dimension が不明で、固定 embedding dimension を事前に決めにくい場合に使う。

## BAxUS の使い方

```python
state = BAxUSState(
    dim=bounds.shape[-1],
    eval_budget=100,
    new_bins_on_split=3,
)
strategy = BAxUSStrategy(bounds, state=state, seed=0)
result = strategy.optimize(acq_function)
y_new = objective(result.candidates)
state = strategy.update_state(
    y_new,
    target_candidates=result.metadata["target_candidates"],
)

if state.restart_triggered:
    strategy.expand_subspace()
```

`BAxUSState` は ambient dimension `dim` と初期設計後に残っている `eval_budget` から、初期 target dimension と expansion schedule を決める。初期 target dimension を利用者が直接指定する方式は廃止した。`new_bins_on_split` も state が一元管理する。

BoTorch の BAxUS tutorial と同じ式で `n_splits`、初期 target dimension、`split_budget`、`failure_tolerance` を導出する。`split_budget` は式の丸め結果をそのまま保持し、小さい評価予算で 0 になる場合も人工的に 1 へ切り上げない。一方 `failure_tolerance` は tutorial と同様に最低 1 へ制限される。target space が拡張されるとこれらは現在の target dimension に対して再計算され、full dimension に到達した場合は `failure_tolerance == dim` となる。これらの式は複数の ambient dimension / evaluation budget に対する conformance test で固定する。

`BAxUSStrategy` は目的関数を評価しない。目的関数評価後に、観測値と `SearchResult.metadata["target_candidates"]` を `update_state()` へ明示的に戻す。target-space の候補と目的値は strategy 内に履歴として保持され、最良の target-space 観測点が次回 trust region の中心になる。q-batch の場合も batch 内の各 target candidate と対応する観測値を同じ履歴へ保存する。

初期 embedding では、各元入力次元を target-space のちょうど 1 bin に割り当て、係数を `+1` または `-1` とする sparse signed embedding を使う。bin の占有数は可能な限り均等化する。

`new_bins_on_split` は BAxUS の 1 回の expansion で各 parent bin から追加できる child bin 数を表す。`expand_subspace()` はメンバーを 2 個以上持つすべての parent bin を対象とし、それぞれを最大 `new_bins_on_split + 1` group に分割する。分割後は `target_dim` を更新し、trust-region length と success / failure counter を初期状態へ戻す。一方、`best_value` と評価予算設定は保持する。

分割時には各元入力次元の `+1/-1` sign と「必ず 1 bin のみに所属する」という sparse embedding の構造を保持する。既存の target-space 観測については parent coordinate を child bin へ複製するため、expansion 前後で元入力空間への projection が保存される。このため expansion 前の探索空間を細分化する nested subspace として扱える。旧 `initial_target_dim` / strategy 側 `new_bins_on_split` API、旧 `new_dimensions` API は残していない。

### lengthscale-weighted trust region

BoTorch の BAxUS tutorial は target-space GP の ARD lengthscale を使って trust-region 各辺をスケーリングする。robotorchan は search strategy と surrogate model を分離し、共通 benchmark では original-space surrogate を維持するため、別の target-space GP を strategy 内で再学習しない。

代わりに acquisition model が original-space ARD lengthscale を公開している場合、現在の sparse embedding が誘導する target-space metric を計算する。target coordinate `j` に所属する元次元集合を `B_j`、元空間 lengthscale を `l_i` とすると、effective target lengthscale は `1 / sqrt(sum_{i in B_j} 1 / l_i^2)` とする。これは ARD の距離 metric を sparse signed embedding 上へ制限したときの target coordinate のスケールに対応する。その後、BoTorch tutorial と同じく重みの幾何平均が 1 になるよう正規化し、`center ± weight * state.length` で trust-region box を作る。

acquisition model から適切な original-space ARD lengthscale を取得できない場合は等方重みへフォールバックする。実際に使った `target_lengthscale_weights` と `target_bounds` は `SearchResult.metadata` に保存する。これは search strategy / surrogate 分離を維持するための設計であり、BoTorch tutorial の「target-space GP を毎反復で fit してその lengthscale を直接使う」実装と完全に同一ではない。

### Thompson sampling candidate generation

`BAxUSThompsonSamplingStrategy` は BoTorch の BAxUS tutorial にある TS 系 candidate generation を search strategy として分離した実装である。現在の target-space trust region 内へ Sobol 点を生成し、各候補について target dimension ごとに `min(20 / target_dim, 1)` の確率で incumbent から perturb する。どの次元も perturb されなかった候補には最低 1 次元の perturbation を強制する。

生成した target-space pool は既存の sparse embedding で original space へ射影し、共通 surrogate model に対する `MaxPosteriorSampling` で候補を選ぶ。したがって TS 用に別の target-space GP を strategy 内で学習しない。選択された original-space candidate に対応する target coordinate は `SearchResult.metadata["target_candidates"]` に戻されるため、通常の BAxUS と同じ `update_state()` / `expand_subspace()` ループを使用できる。

```python
strategy = BAxUSThompsonSamplingStrategy(
    bounds,
    state=state,
    seed=0,
    n_candidates=5000,
)
result = strategy.optimize(acq_function, q=1)
```

`acq_function` は候補選択そのものではなく、共通 model の取得と返却候補の acquisition value 診断に利用する。候補選択は posterior sample に基づく。`n_candidates` は TS の有限候補 pool サイズであり、`q` を下回ってはならない。

## 設計上の注意

- search strategy は surrogate を置き換えない。
- acquisition function には最終的に元入力空間の候補を渡す。
- `SearchResult` は候補点・獲得関数値・探索診断だけを返し、wall-clock 時間は保持しない。
- 探索時間は benchmark / 呼び出し側で `strategy.optimize(...)` の前後を計測する。
- `LatentSpaceStrategy` はデータから学習した reducer、REMBO / HeSBO / BAxUS は探索用の埋め込みであり役割が異なる。
- REMBO は dense Gaussian embedding、HeSBO は fixed sparse hash embedding、BAxUS は適応的に拡張する sparse signed embedding である。
- TuRBO / BAxUS は stateful なので、逐次 BO ループ側が観測値を state に戻す必要がある。
- BAxUS の trust region は最良 target observation を中心とし、利用可能な場合は original-space ARD metric から誘導した target-space weight で各辺をスケーリングする。
- target-space GP を strategy 内で別途 fit しないため、BoTorch tutorial と完全に同じ surrogate 構成ではない。surrogate を全探索戦略で共通化する robotorchan の責務分離を優先している。

## ベンチマーク

`benchmarks/high_dimensional_acqf_optimization.py` は連続獲得関数最適化そのものを比較する。`OriginalSpace`、`RandomSearch`、`REMBO`、`HeSBO` は同一の fitted original-space `SingleTaskGP` と同一の `PosteriorMean` に対して比較し、モデル差ではなく search strategy の差を測る。REMBO と HeSBO には同じ `embedding_dim` を与える一方、problem generation、RandomSearch、REMBO、HeSBO の乱数 stream は互いに分離する。

`benchmarks/high_dimensional_sequential_bo.py` は逐次 BO の simple regret と探索時間を比較する。固定 embedding の `REMBO` と `HeSBO` は strategy 生成時に embedding を固定し、BO iteration ごとに再生成しない。両者とも original-space `SingleTaskGP` と各反復の `LogExpectedImprovement` を共有するため、逐次 BO でも surrogate の違いではなく search strategy の違いとして比較する。
モデル性能と探索戦略性能を混同しないため、探索戦略比較では同じ surrogate / acquisition を使うことを原則とする。
BAxUS には `n_iterations` を初期設計後の評価予算として渡し、target dimension を評価予算から導出する。
探索時間はライブラリの戻り値ではなく、benchmark 側で外部計測する。


## dtype / device contract

Search strategy は public-space `bounds` の dtype / device を保持する。strategy 内部で生成する random sample、embedding、latent / target bounds、diagnostic tensor は原則として `bounds` と同じ dtype / device 上に置き、`SearchResult.candidates` を暗黙に CPU や別 dtype へ変換しない。

BoTorch の数値安定性を考慮し、通常の GP ベース BO では `torch.float64` を推奨する。GPU を使用する場合も、model / acquisition / bounds を同一 device に配置する。CPU CI では float64 の dtype/device propagation を常時検証し、CUDA が利用可能な環境では RandomSearch の device propagation test も実行する。


### ALEBO原実装との対応

ALEBOの公開参照実装では、MAP推定後にMahalanobis kernelのCholeskyパラメータだけをLaplace近似でサンプリングし、mean constantとoutput scaleはMAP値に固定する。さらにmetric sampleの先頭にはMAP値そのものを含める。robotorchanもmetric uncertaintyについてこのMAP-first sampling規約を採用する。なお、現実装はBoTorchの現行APIに合わせた構成であり、旧Axクラスの互換APIは提供しない。


### Multi-fidelity candidate constraints

`OriginalSpaceStrategy` can combine `CandidateConstraints` with `fixed_features`. A target fidelity can therefore be fixed with, for example, `fixed_features={fidelity_dim: 1.0}` while linear equality or inequality constraints remain expressed in the same raw/public input coordinates. This delegates both mechanisms to BoTorch `optimize_acqf`; output/black-box constraints remain a separate acquisition-level concern.


### Constraints with embedding and latent search

Candidate constraints are defined in public/original input coordinates. They must not be forwarded unchanged to an optimizer operating in REMBO, HeSBO, BAxUS, or learned latent coordinates, because the tuple indices and coefficients would then describe a different constraint. These strategies therefore reject non-empty `CandidateConstraints` until an exact strategy-specific mapping is implemented. This is intentional capability validation rather than silent approximate feasibility.

ALEBO is different: its existing linear inequalities describe the **internal ALEBO embedding polytope** that keeps projected candidates inside the public box. Those internal inequalities are not user process constraints and must not be interpreted as general `CandidateConstraints` support.

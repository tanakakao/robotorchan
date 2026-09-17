# 高次元ベイズ最適化の探索戦略

robotorchan では surrogate model と acquisition function と search strategy を分離する。
高次元モデルを変更せず、獲得関数の最適化方法だけを切り替えられる。

## 戦略

| Strategy | 探索空間 | 状態 | 主な用途 |
| --- | --- | --- | --- |
| `OriginalSpaceStrategy` | 元の d 次元 | なし | 標準 `optimize_acqf`、低〜中次元の基準 |
| `RandomSearchStrategy` | 元の d 次元 | なし | 高次元での単純・堅牢なベースライン |
| `LatentSpaceStrategy` | PCA / RandomProjection 潜在空間 | reducer に依存 | 既知の低次元表現を使う探索 |
| `REMBOStrategy` | 固定ランダム埋め込み | 固定 | intrinsic dimension が低いと期待できる場合 |
| `TuRBOStrategy` | 元空間の局所 trust region | あり | 高次元で局所探索を適応的に集中 |
| `BAxUSStrategy` | sparse signed embedding | あり | intrinsic dimension が不明で、探索中に部分空間を拡張したい場合 |
| `BAxUSThompsonSamplingStrategy` | sparse signed embedding | あり | BAxUS の trust region 内で sparse perturbation + Thompson sampling を使う場合 |

### RandomSearch の q-batch

`RandomSearchStrategy` の `num_samples` は候補点数ではなく、評価するランダムな q-batch 数を表す。
`q=1` では `num_samples` 個の点を比較し、`q>1` では shape `[num_samples, q, d]` の候補 batch を生成して、獲得関数が返す joint batch value が最大の 1 batch を返す。
したがって `q>1` でも各点を q=1 として独立評価して上位 q 点を並べる処理は行わない。

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

BoTorch の BAxUS tutorial と同様に、`n_splits`、`split_budget`、`failure_tolerance` を target dimension と評価予算から導出する。`failure_tolerance` は固定値ではなく、target space が拡張されると再計算される。full dimension に到達した場合は `failure_tolerance == dim` となる。

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
- `LatentSpaceStrategy` はデータから学習した reducer、REMBO / BAxUS は探索用の埋め込みであり役割が異なる。
- TuRBO / BAxUS は stateful なので、逐次 BO ループ側が観測値を state に戻す必要がある。
- BAxUS の trust region は最良 target observation を中心とし、利用可能な場合は original-space ARD metric から誘導した target-space weight で各辺をスケーリングする。
- target-space GP を strategy 内で別途 fit しないため、BoTorch tutorial と完全に同じ surrogate 構成ではない。surrogate を全探索戦略で共通化する robotorchan の責務分離を優先している。

## ベンチマーク

`benchmarks/high_dimensional_acqf_optimization.py` は連続獲得関数最適化そのものを比較する。
`benchmarks/high_dimensional_sequential_bo.py` は逐次 BO の simple regret と探索時間を比較する。
モデル性能と探索戦略性能を混同しないため、探索戦略比較では同じ surrogate / acquisition を使うことを原則とする。
BAxUS には `n_iterations` を初期設計後の評価予算として渡し、target dimension を評価予算から導出する。
探索時間はライブラリの戻り値ではなく、benchmark 側で外部計測する。

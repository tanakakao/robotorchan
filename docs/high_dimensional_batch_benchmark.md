# 高次元 batch BO ベンチマーク

`benchmarks/high_dimensional_batch_bo.py` は、joint q-batch の高次元ベイズ最適化を比較するためのベンチマークである。

比較対象は `OriginalSpace`、`RandomSearch`、`REMBO`、`HeSBO`、`TuRBO`、`BAxUS`、`BAxUSTS`。`REMBO` と `HeSBO` には共通の `--embedding-dim` を与え、固定 embedding は batch BO trajectory 全体で再生成しない。全戦略で同じ original-space `SingleTaskGP` を学習し、診断用 acquisition として `qLogExpectedImprovement` を使用する。`BAxUSTS` の候補選択自体は acquisition 最大化ではなく posterior Thompson sampling に基づくため、純粋な acquisition-search 比較ではなく BO policy 比較として扱う。

## BAxUSTS の候補 pool

`BAxUSTS` では `ts_candidates=None` を標準とする。この場合、`BAxUSThompsonSamplingStrategy` が BoTorch BAxUS tutorial と同じ規則

```text
min(5000, max(2000, 200 * target_dim))
```

で候補 pool サイズを決める。ここで `target_dim` は現在の BAxUS target-space 次元であり、ambient input 次元ではない。BoTorch tutorial の `X.shape[-1]` は target-space 観測 `X_baxus_target` の次元を参照する。したがって subspace expansion 後は自動候補数も現在の `target_dim` に追従する。例えば target 次元が 2 / 15 / 30 の場合はそれぞれ 2000 / 3000 / 5000 となる。

再現性や計算量を固定した実験では `--ts-candidates` を明示指定できる。明示値は `q` 以上でなければならない。旧来の「benchmark 側で常に 5000 を渡す」挙動は残していない。

## 評価予算

BAxUS 系には `n_iterations * q` を初期設計後の評価予算として渡す。batch 内の全 candidate と objective value を target-space history へ戻し、restart 条件を満たした場合は nested sparse embedding を拡張する。

探索時間は `SearchResult` ではなく benchmark 側で `strategy.optimize(...)` の前後を計測する。

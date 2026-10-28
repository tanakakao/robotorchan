# Benchmark の保存と再開

次の条件で、単一 seed の checkpoint と決定的な継続に対応します。

```python
from robotorchan.benchmarks import (
    load_benchmark_checkpoint,
    run_resumable_benchmark,
    save_benchmark_checkpoint,
)

partial, checkpoint = run_resumable_benchmark(
    config, random_candidates, registry=registry, max_batches=2
)
save_benchmark_checkpoint("checkpoint.json", checkpoint)

restored = load_benchmark_checkpoint("checkpoint.json")
finished, _ = run_resumable_benchmark(
    config, random_candidates, registry=registry, checkpoint=restored
)
```


checkpoint は軌跡の prefix、実験設定、run-local の PyTorch generator 状態を version 付き JSON で保存し、読み込みで Python オブジェクトを実行しません。1 checkpoint は設定済み seed 1 個のみを扱います。完了済み checkpoint を再度渡しても新規評価を行いません。

履歴と run-local RNG の純粋関数である候補 generator なら厳密に継続できます。内部状態、global RNG、optimizer 状態、pending 非同期評価、非決定的ノイズはこの契約だけでは厳密に復元できません。保存位置は batch 完了後で、batch の途中ではありません。呼び出し側が返された checkpoint を明示的に永続化し、自動定期保存や atomic writer は提供しません。

完了結果の[保存形式](storage.md)とは別 schema です。結果ファイルだけでは継続に必要な generator 状態を保持できません。

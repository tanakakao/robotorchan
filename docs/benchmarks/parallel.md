# 逐次・batch・非同期の比較

`run_parallel_comparison(config, batch_size=3, max_concurrency=2)` は同じ問題・seed・初期設計・候補評価予算で Random の 3 経路を比較します。

- 逐次：q=1、次の提案前に観測を反映。
- 同期 batch：q=batch_size、batch ごとに観測更新。
- 非同期：q=batch_size、並列数を制限し、完了観測を即時反映。未完了候補は pending。

非同期では候補ごとに正の固定 duration 1 を使います。`simulated_makespan` は論理時間であり実時間ではありません。`completion_order` は完了順の投入 index を記録し、pending と観測値・真値を分離します。

比較対象は実行の意味で、optimizer の品質ではありません。3 経路は評価数を揃えますが、提案更新数や論理経過時間は揃いません。初期設計は共通です。非同期 runner も seed ごとに factory を再生成し、状態付き観測 RNG の漏れを防ぎます。

# Phase 16: acquisition benchmark の RNG 分離

`benchmarks/high_dimensional_acqf_optimization.py` では、初期学習データ生成と RandomSearch の候補生成に同じ乱数 seed を直接渡さないようにします。

以前は両方に同じ seed を使っていたため、同じ `torch.rand` のストリーム先頭を利用し、RandomSearch の候補集合が学習点と重なる可能性がありました。これは探索戦略の比較として不適切です。

Phase 16 では、実験を識別する `seed` はそのまま初期データ生成に使い、RandomSearch には決定的に導出した別の search seed を使います。これにより再現性を維持したまま、problem generation と acquisition search の乱数ストリームを分離します。

ベンチマーク結果の `seed` 列は従来どおり実験 seed を表します。search seed は内部実装上の乱数分離にのみ使用します。

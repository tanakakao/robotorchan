# 高次元 Sequential BO ベンチマーク

`benchmarks/high_dimensional_sequential_bo.py` は、高次元連続空間における **獲得関数の探索戦略**を比較するためのベンチマークです。

比較対象間で surrogate model は元入力空間の `SingleTaskGP` に統一します。したがって、PCA や Random Projection を使う戦略でも surrogate 自体は次元削減せず、探索空間だけを低次元化します。

獲得関数には q=1 の `LogExpectedImprovement` を使用します。各 iteration で現在の観測値
`train_Y.max()` を `best_f` として更新するため、posterior mean の最大化だけを行う greedy benchmark ではなく、改善量と不確実性を考慮する sequential Bayesian optimization の比較になります。

PCA / Random Projection の search reducer は初期設計点から一度だけ学習し、その BO trajectory 中は固定します。これにより iteration ごとの座標系変更を避け、SearchStrategy の差を比較しやすくしています。

現在のベンチマークは q=1 専用です。q>1 の batch BO を比較する場合は `qLogExpectedImprovement` 等を使う別設定として扱います。

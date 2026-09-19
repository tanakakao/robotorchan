# 高次元 Sequential BO ベンチマーク

`benchmarks/high_dimensional_sequential_bo.py` は、高次元連続空間における **獲得関数の探索戦略**を比較するためのベンチマークです。

比較対象間で surrogate model は元入力空間の `SingleTaskGP` に統一します。したがって、PCA や Random Projection を使う戦略でも surrogate 自体は次元削減せず、探索空間だけを低次元化します。

獲得関数には q=1 の `LogExpectedImprovement` を使用します。各 iteration で現在の観測値 `train_Y.max()` を `best_f` として更新するため、posterior mean の最大化だけを行う greedy benchmark ではなく、改善量と不確実性を考慮する sequential Bayesian optimization の比較になります。

比較対象は `OriginalSpace`、`RandomSearch`、`LatentPCA`、`LatentRandomProjection`、`REMBO`、`TuRBO`、`BAxUS` です。PCA / Random Projection の search reducer と REMBO の random embedding は trajectory の開始時に一度だけ作成して固定します。

TuRBO と BAxUS は strategy instance を iteration 間で保持します。TuRBO は初期観測中の best point を original-space incumbent として開始し、新しい objective observation に応じて trust-region state と incumbent を更新します。BAxUS も初期 best value から state を開始し、各 observation 後に state を更新します。restart 条件を満たし、まだ入力次元まで到達していない場合は subspace を拡張します。

`RandomSearch` だけは各 iteration で異なる deterministic seed を使って候補集合を再生成します。同じ random candidate set を毎回再利用しないためです。

現在のベンチマークは q=1 専用です。q>1 の batch BO を比較する場合は `qLogExpectedImprovement` 等を使う別設定として扱います。

## 再現性と解釈

この benchmark の主目的は、同一 surrogate と同一 objective の下で candidate search strategy の差を比較することです。したがって結果を surrogate model 自体の優劣として解釈しません。

比較時には input dimension、seed、初期設計、評価回数、objective、獲得関数を揃えます。PCA / Random Projection / REMBO の固定写像を trajectory 途中で再学習・再生成すると、search strategy 以外の差が混入するため固定します。

実行コードと CLI の現在仕様はリポジトリの `benchmarks/high_dimensional_sequential_bo.py` を正とします。実装変更時は本書の strategy 一覧、q 契約、stateful strategy の説明も同時に更新してください。

# Benchmark scripts

このディレクトリには robotorchan の比較・回帰確認に使う実行スクリプトを置きます。

高次元 Bayesian optimization の実験条件と結果の解釈契約は [docs/benchmarks/high_dimensional](../docs/benchmarks/high_dimensional/README.md) を参照してください。

`src/robotorchan/benchmarks/` はライブラリから再利用する構造的 benchmark API、
このルート `benchmarks/` は手動・実験用の executable script という責務で分離します。

主なスクリプト:

- `high_dimensional_sequential_bo.py`: q=1 sequential BO
- `high_dimensional_batch_bo.py`: joint q-batch BO
- `high_dimensional_bo.py`: 高次元モデル比較
- `high_dimensional_inputs.py`: 入力次元に対する predictive benchmark
- `high_dimensional_multitask.py`: multi-task 高次元 benchmark
- `high_dimensional_saas.py`: SAAS 系 benchmark
- `high_dimensional_acqf_optimization.py`: acquisition optimization benchmark
- `expressive_predictive.py`: expressive surrogate predictive benchmark
- `non_gp_surrogates.py`: empirical non-GP surrogate predictive diagnostics

実装の正は各スクリプトと `tests/benchmarks/` です。文書側には、再現に必要な条件と比較結果を誤解しないための境界を残します。

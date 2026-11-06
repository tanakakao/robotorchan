# Benchmark scripts

このディレクトリには robotorchan の比較・回帰確認に使う実行スクリプトを置きます。

高次元 Bayesian optimization の実験条件と結果の解釈契約は [docs/benchmarks/high_dimensional](../docs/benchmarks/high_dimensional/README.md) を参照してください。

`src/robotorchan/benchmarks/` はライブラリから再利用する構造的 benchmark API、
このルート `benchmarks/` は手動・実験用の executable script という責務で分離します。

ルート `benchmarks/` は importable package ではありません。再利用する定義・metric・scenario は
`robotorchan.benchmarks` に置き、このディレクトリのファイルは script path から実行します。

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
- `q_epig_batch_comparison.py`: marginal / joint / pending-aware qEPIG の同一予算比較\n- `turbo.py`: Ackley / Rosenbrock / Levy で Global BO と TuRBO を比較する再現可能 benchmark

実装の正は各スクリプトと `tests/benchmarks/` です。文書側には、再現に必要な条件と比較結果を誤解しないための境界を残します。

## qEPIG batch comparison

```bash
uv run python benchmarks/q_epig_batch_comparison.py --seed 0 --q 2
```

各手法について獲得関数値、同一の joint EPIG / pending 条件付き EPIG、候補間最小距離、
最適化時間を JSON Lines 形式で出力します。獲得関数値そのものは目的関数が異なるため、
手法間で直接比較しないでください。候補間距離は多様性の診断値であり、最適化性能の証明ではありません。
単一 seed の例は動作確認用です。統計的な優劣の主張には複数 seed と反復評価が必要です。

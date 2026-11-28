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


## Known-DAG Causal BO benchmark

Run `python benchmarks/causal_bo.py --seeds 0,1,2 --iterations 5`
to compare CBO, fixed-full-intervention GP-BO and Random Search from
identical initial hard-intervention experiments. Per-evaluation CSV
and seed-aggregated JSON outputs include best outcome, simple regret,
and cumulative intervention cost. Interpretation and limitations are
in [docs/benchmarks/README.md](../docs/benchmarks/README.md).


## Decoupled Bayesian optimization

同一の初期観測と累積評価予算で、全出力を測定する Coupled BO と部分観測を選択する
Decoupled BO を比較します。各スクリプトは JSON Lines を標準出力に書き出します。

| スクリプト | 対象 |
| --- | --- |
| `decoupled_bo.py` | Branin-Currin、ZDT1、DTLZ2、EPIG/KG/HVKG/Random/Sobol |
| `decoupled_constrained_bo.py` | 制約付き回帰、Mixed Process Yield |
| `decoupled_multifidelity_bo.py` | 離散 fidelity と部分観測 |
| `decoupled_heterogeneous_bo.py` | 回帰と二値・多クラス・順序分類 |
| `decoupled_materials_bo.py` | 強度・導電率・品質合否の材料探索模擬問題 |
| `decoupled_report.py` | コスト整列、Bootstrap、比較グラフ |

学習を含む最小実行例（初期コスト16、総予算20）:

```bash
uv run python benchmarks/decoupled_bo.py --problem branin_currin --seeds 0,1 --budget 20 --policies kg > decoupled.jsonl
uv run --extra examples python benchmarks/decoupled_report.py decoupled.jsonl --metric hypervolume --start 16 --stop 20 --plot decoupled.png
```

材料探索の例:

```bash
uv run python benchmarks/decoupled_materials_bo.py --seeds 0,1,2 --weights 0.5 --budget 60 > materials.jsonl
uv run --extra examples python benchmarks/decoupled_report.py materials.jsonl --metric utility --start 36 --stop 60 --baseline coupled_particle_kg --plot materials.png
```

- 初期観測も予算に含めます。残予算で評価できない場合は終了します。
- 共通コスト格子では直前の推奨結果を保持し、未来の観測値を補間しません。
  終了後の保持は追加観測がない場合の性能を表します。中断した実験とは区別してください。
- Bootstrap は seed 軌跡単位で再標本化する点ごとの区間です。同時信頼帯ではありません。
  少数 seed は動作確認用です。比較には十分な独立 seed と同一条件が必要です。
- 欠損した regret をゼロに置換しません。制約付き問題では feasibility を別に報告します。
- KG と材料探索の効用は固定重みのスカラー化です。HVKG の Pareto 最適化とは区別します。
- 模擬真値は評価指標の算出にも使用しますが、学習には選択された出力のみを追加します。
- Multi-Fidelity のコスト、分類の効用、Particle KG の粒子数は実験設定であり、
  実測材料データに対する性能保証ではありません。

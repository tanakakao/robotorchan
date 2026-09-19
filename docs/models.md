# robotorchan モデル概要・使い所ガイド

> 対象: `robotorchan.models` の現行 public API  
> 方針: BoTorch のモデル挙動を維持しつつ、raw data の保持や `make_mll()` など robotorchan 共通 API を追加する。

## 1. このドキュメントの目的

このドキュメントは「どのモデルを選ぶか」を判断するためのガイドです。実際の構築・学習・事後分布予測は、各モデルに対応する Jupyter Notebook を参照してください。

- モデル選択: この `docs/models.md`
- 実行例: [`examples/notebooks/`](../examples/notebooks/)
- Notebook 一覧: [`examples/README.md`](../examples/README.md)
- 実装設計: [`docs/architecture.md`](architecture.md)
- 正しさの検証: `tests/`
- Public model のドキュメント対応表: [`docs/model_coverage.json`](model_coverage.json)

## 2. モデル選択早見表

| 状況 | 第一候補 | 実行例 |
|---|---|---|
| 通常の連続変数 + 単一目的 | `SingleTaskGP` | [01](../examples/notebooks/01_single_task_gp.ipynb) |
| 連続変数 + カテゴリ変数 | `MixedSingleTaskGP` | [02](../examples/notebooks/02_mixed_single_task_gp.ipynb) |
| 低精度/高精度データを併用 | `SingleTaskMultiFidelityGP` | [03](../examples/notebooks/03_multi_fidelity_gp.ipynb) |
| 複数タスクで情報共有 | `MultiTaskGP` | [04](../examples/notebooks/04_multitask_gp.ipynb) |
| 全タスクを同じ X で測定 | `KroneckerMultiTaskGP` | [04](../examples/notebooks/04_multitask_gp.ipynb) |
| 出力ごとに独立 GP | `ModelListGP` | [05](../examples/notebooks/05_model_list_gp.ipynb) |
| Exact GP が重い大規模データ | `SingleTaskVariationalGP` | [06](../examples/notebooks/06_variational_gp.ipynb) |
| 比較・選好データ | `PairwiseGP` | [07](../examples/notebooks/07_pairwise_gp.ipynb) |
| 高次元・少数の有効変数 | `SaasFullyBayesianSingleTaskGP` | [08](../examples/notebooks/08_saas_gp.ipynb) |
| 高次元・マルチタスク | `SaasFullyBayesianMultiTaskGP` | [08](../examples/notebooks/08_saas_gp.ipynb) |
| 高次元・MAP-SAAS | `AdditiveMapSaasSingleTaskGP` / `EnsembleMapSaasSingleTaskGP` | [09](../examples/notebooks/09_map_saas_and_additive_gp.ipynb) |
| 加法構造を仮定 | `OrthogonalAdditiveGP` | [09](../examples/notebooks/09_map_saas_and_additive_gp.ipynb) |
| 外れ値の影響を抑えたい | `RobustRelevancePursuitSingleTaskGP` | [10](../examples/notebooks/10_robust_gp.ipynb) |
| 裾の重い観測ノイズを扱いたい | `StudentTSingleTaskGP` | [15](../examples/notebooks/15_robust_observation_models.ipynb) |
| 一部の観測が汚染分布に由来する | `ContaminatedSingleTaskGP` | [15](../examples/notebooks/15_robust_observation_models.ipynb) |
| 入力位置で観測ノイズが変わる | `HeteroskedasticSingleTaskGP` / `JointHeteroskedasticSingleTaskGP` | [16](../examples/notebooks/16_noise_models.ipynb) |
| 同一条件の反復測定からノイズを推定 | `ReplicateNoiseSingleTaskGP` | [16](../examples/notebooks/16_noise_models.ipynb) |
| 入力空間で滑らかさが変化する | `NonstationarySingleTaskGP` | [18](../examples/notebooks/18_nonstationary_gp.ipynb) |
| 連続入力自体に測定不確かさがある | `UncertainInputSingleTaskGP` | [17](../examples/notebooks/17_uncertain_input_gp.ipynb) |
| カテゴリ入力に確率的不確かさがある | `UncertainCategoricalSingleTaskGP` | [17](../examples/notebooks/17_uncertain_input_gp.ipynb) |
| テンソル出力 | `HigherOrderGP` | [11](../examples/notebooks/11_structured_output_gp.ipynb) |
| 時間・波長・位置など明示的な出力軸 | `LatentKroneckerGP` | [11](../examples/notebooks/11_structured_output_gp.ipynb) |
| 条件によって有効変数が変わる | `HierarchicalConditionalKernelGP` | [12](../examples/notebooks/12_hierarchical_gp.ipynb) |
| 階層探索 + マルチタスク | `HierarchicalConditionalKernelMultiTaskGP` | [12](../examples/notebooks/12_hierarchical_gp.ipynb) |
| タスクごとに特徴量構造が異なる | `HeterogeneousMTGP` | [13](../examples/notebooks/13_heterogeneous_multitask_gp.ipynb) |
| Context ごとの加法構造 | `SACGP` | [14](../examples/notebooks/14_contextual_gp.ipynb) |
| Context 間の潜在関係を学習 | `LCEAGP` | [14](../examples/notebooks/14_contextual_gp.ipynb) |
| Context / Task を multi-output として扱う | `LCEMGP` | [14](../examples/notebooks/14_contextual_gp.ipynb) |

## 3. robotorchan 共通 API

多くの supervised wrapper は、BoTorch のモデル機能に加えて次の情報を保持します。

```python
model.raw_train_X
model.raw_train_Y
model.raw_train_Yvar
model.raw_data
model.raw_data_names
model.supports_mll
model.make_mll()
```

`raw_*` は **コンストラクタへ渡したデータのスナップショット** です。`condition_on_observations()` や `fantasize()` 後の現在状態を表すものではありません。最新の学習状態は BoTorch の `train_inputs` / `train_targets` を参照してください。

Exact GP 系の基本的な学習は次の形です。

```python
from botorch.fit import fit_gpytorch_mll

mll = model.make_mll()
fit_gpytorch_mll(mll)
```

モデルによっては学習方法が異なります。たとえば Variational GP は `VariationalELBO`、Fully Bayesian SAAS は NUTS を使います。

## 4. 標準モデル

### `SingleTaskGP`

連続変数中心の単一タスク回帰に使う基本モデルです。迷った場合の最初のベースラインとして適しています。

材料・製造では温度、圧力、時間、組成比、連続プロセス条件などを入力として性能を予測する通常の BO に向きます。

高次元で有効変数が少ない場合は SAAS 系も検討します。

**Notebook:** [01_single_task_gp.ipynb](../examples/notebooks/01_single_task_gp.ipynb)

### `MixedSingleTaskGP`

連続変数とカテゴリ変数が混在する探索空間向けです。装置種、製法、触媒種、材料種などをカテゴリ次元として明示できます。

カテゴリ値を単純な連続値として `SingleTaskGP` に入れる代替ではありません。`cat_dims` を指定します。

**Notebook:** [02_mixed_single_task_gp.ipynb](../examples/notebooks/02_mixed_single_task_gp.ipynb)

## 5. Multi-Fidelity

### `SingleTaskMultiFidelityGP`

低 fidelity の安価な情報と高 fidelity の高精度情報を共有して学習します。シミュレーション精度、メッシュ解像度、簡易試験 / 実機試験などに向きます。

低 fidelity と高 fidelity に相関があり、高 fidelity の評価コストが高いときに特に有効です。

**Notebook:** [03_multi_fidelity_gp.ipynb](../examples/notebooks/03_multi_fidelity_gp.ipynb)

## 6. Multi-task / Multi-output

### `MultiTaskGP`

BoTorch の long-format 表現を使い、タスク間相関を学習します。タスクごとに異なる X で観測されていても扱えます。

装置、製品、測定方法などを「関連する別タスク」として情報共有したい場合に向きます。

**Notebook:** [04_multitask_gp.ipynb](../examples/notebooks/04_multitask_gp.ipynb)

### `KroneckerMultiTaskGP`

すべての X で全タスクが観測される block design 向けです。

```text
train_X: [n, d]
train_Y: [n, m]
```

同じ条件で複数特性を毎回測定する場合に適しています。

**Notebook:** [04_multitask_gp.ipynb](../examples/notebooks/04_multitask_gp.ipynb)

### `ModelListGP`

独立した複数 GP を1つの BoTorch model としてまとめます。出力間共分散を学習する必要がない場合や、多目的 BO で目的ごとに別 GP を持つ場合に自然です。

複数目的だから必ず `MultiTaskGP` にする必要はありません。

**Notebook:** [05_model_list_gp.ipynb](../examples/notebooks/05_model_list_gp.ipynb)

## 7. 大規模データ

### `SingleTaskVariationalGP`

inducing point を使う近似 GP です。Exact GP の計算量がボトルネックになるデータ量で利用します。

`make_mll()` は `VariationalELBO` を返し、minibatch 学習時は全データ件数を `num_data` として扱います。

**Notebook:** [06_variational_gp.ipynb](../examples/notebooks/06_variational_gp.ipynb)

## 8. Preference Learning

### `PairwiseGP`

絶対スコアではなく「A の方が B より良い」という比較データを学習します。官能評価、デザイン選好、ランキング、定量化しにくい品質評価などに向きます。

通常の `train_Y` ではなく `datapoints` と `comparisons` を使います。

**Notebook:** [07_pairwise_gp.ipynb](../examples/notebooks/07_pairwise_gp.ipynb)

## 9. 高次元モデル

### `SaasFullyBayesianSingleTaskGP`

SAAS prior と NUTS を使う fully Bayesian GP です。高次元空間で少数の次元のみが重要だと考えられる場合に強力です。

計算コストは高く、`supports_mll=False` です。

**Notebook:** [08_saas_gp.ipynb](../examples/notebooks/08_saas_gp.ipynb)

### `SaasFullyBayesianMultiTaskGP`

SAAS の multi-task 版です。高次元性とタスク間共有の両方を扱います。

**Notebook:** [08_saas_gp.ipynb](../examples/notebooks/08_saas_gp.ipynb)

### `AdditiveMapSaasSingleTaskGP`

MAP 推定を使う SAAS 系モデルです。Fully Bayesian SAAS より軽量な高次元モデル候補です。

**Notebook:** [09_map_saas_and_additive_gp.ipynb](../examples/notebooks/09_map_saas_and_additive_gp.ipynb)

### `EnsembleMapSaasSingleTaskGP`

複数の MAP-SAAS 設定を ensemble として扱います。NUTS を避けつつ SAAS 的な高次元モデリングを行いたい場合に候補になります。

**Notebook:** [09_map_saas_and_additive_gp.ipynb](../examples/notebooks/09_map_saas_and_additive_gp.ipynb)

### `OrthogonalAdditiveGP`

1次・低次の加法構造を仮定する高次元 GP です。目的関数が少数変数の加法的な寄与で近似できると考えられる場合に向きます。

**Notebook:** [09_map_saas_and_additive_gp.ipynb](../examples/notebooks/09_map_saas_and_additive_gp.ipynb)

### `RobustRelevancePursuitSingleTaskGP`

少数の異常観測や強い外れ値の影響を抑えたい場合の robust GP です。

外れ値に見えるデータが実際には別レジームなら、階層・タスク・混合モデルの方が適切なことがあります。

**Notebook:** [10_robust_gp.ipynb](../examples/notebooks/10_robust_gp.ipynb)

## 10. Robust / Noise / Input Uncertainty

通常の Gaussian observation noise では表現しにくい外れ値、入力依存ノイズ、反復測定、入力不確かさ、非定常性には専用モデルを使います。これらは同じ「robust GP」ではなく、**どの仮定が通常 GP から外れているか**で選択します。

### 外れ値・汚染観測

- `RobustRelevancePursuitSingleTaskGP`: 少数の異常観測を relevance pursuit で扱います。
- `StudentTSingleTaskGP`: heavy-tailed な Student-t 観測モデルで大きな残差への感度を抑えます。
- `ContaminatedSingleTaskGP`: 通常観測と汚染観測の mixture として外れ値をモデル化します。

詳細: [Student-t GP](models/student_t_gp.md)、[Contaminated GP](models/contaminated_gp.md)

### Heteroskedastic / replicate noise

- `HeteroskedasticSingleTaskGP`: 応答 GP と推定された入力依存ノイズ過程を反復的に扱います。
- `JointHeteroskedasticSingleTaskGP`: 応答と log-noise の潜在過程を joint に学習します。
- `ReplicateNoiseSingleTaskGP`: 同一入力の反復測定から group mean と variance-of-the-mean を構成します。

詳細: [Heteroskedastic feasibility](models/heteroskedastic_gp_feasibility.md)、[Joint heteroskedastic GP](models/joint_heteroskedastic_gp.md)、[Replicate-noise GP](models/replicate_noise_gp.md)

### Input uncertainty

- `UncertainInputSingleTaskGP`: 連続 training input の Gaussian uncertainty を covariance に反映します。
- `UncertainCategoricalSingleTaskGP`: categorical input の不確かさをカテゴリ確率として扱います。
- `MixedUncertainInputSingleTaskGP`: continuous uncertainty と deterministic nominal category を分離して扱います。

カテゴリコードを連続量として Gaussian perturbation するモデルではありません。

詳細: [Uncertain-input GP](models/uncertain_input_gp.md)、[Uncertain categorical GP](models/uncertain_categorical_gp.md)、[Mixed uncertain-input GP](models/mixed_uncertain_input_gp.md)

### Nonstationary GP

`NonstationarySingleTaskGP` は入力位置によって局所 lengthscale が変化する場合に使います。観測ノイズが場所によって変化する heteroskedastic GP とは対象とする非定常性が異なります。

詳細: [Nonstationary GP](models/nonstationary_gp.md)

### Mixed robust / noise models

robotorchan には continuous 版の単純な alias ではなく、native categorical covariance を使う Mixed 実装があります。

- `MixedStudentTSingleTaskGP`
- `MixedContaminatedSingleTaskGP`
- `MixedHeteroskedasticSingleTaskGP`
- `MixedJointHeteroskedasticSingleTaskGP`
- `MixedReplicateNoiseSingleTaskGP`
- `MixedNonstationarySingleTaskGP`
- `MixedUncertainInputSingleTaskGP`
- `MixedRobustRelevancePursuitSingleTaskGP`

設計上の対応範囲は [Robust × Mixed coverage](robust-mixed-coverage-audit.md) を参照してください。

## 11. 構造化出力

### `HigherOrderGP`

画像、スペクトルグリッド、空間分布など tensor-valued output を直接扱うモデルです。

出力テンソルの各軸そのものに構造的意味がある場合に向きます。

**Notebook:** [11_structured_output_gp.ipynb](../examples/notebooks/11_structured_output_gp.ipynb)

### `LatentKroneckerGP`

入力 `X` に加えて、時間・波長・位置などの出力座標 `T` を明示的に扱います。robotorchan は `raw_train_T` も保持します。

新しい `T` グリッドで補間・予測したい場合に特に自然です。

**Notebook:** [11_structured_output_gp.ipynb](../examples/notebooks/11_structured_output_gp.ipynb)

## 12. 階層・条件付き探索空間

### `HierarchicalConditionalKernelGP`

親変数の値によって有効になる子変数が変わる探索空間向けです。

例:
- 装置方式 A では温度が有効
- 装置方式 B では圧力が有効

通常の mixed model では表しにくい「条件付きで存在する変数」を扱えます。

**Notebook:** [12_hierarchical_gp.ipynb](../examples/notebooks/12_hierarchical_gp.ipynb)

### `HierarchicalConditionalKernelMultiTaskGP`

上記に task covariance を加えた multi-task 版です。

**Notebook:** [12_hierarchical_gp.ipynb](../examples/notebooks/12_hierarchical_gp.ipynb)

## 13. Heterogeneous Multi-task

### `HeterogeneousMTGP`

タスクごとに入力特徴量の構成が異なる場合に使います。

各タスクで別々の `train_Xs` / `train_Ys` を持ち、`feature_indices` で共通の full feature space へ対応付けます。

**Notebook:** [13_heterogeneous_multitask_gp.ipynb](../examples/notebooks/13_heterogeneous_multitask_gp.ipynb)

## 14. Contextual GP

### `SACGP`

入力を context ごとの成分へ分解し、構造的な加法関係を利用する Contextual GP です。

**Notebook:** [14_contextual_gp.ipynb](../examples/notebooks/14_contextual_gp.ipynb)

### `LCEAGP`

context の潜在 embedding を学習し、context 間の関係もモデル化する additive contextual GP です。

**Notebook:** [14_contextual_gp.ipynb](../examples/notebooks/14_contextual_gp.ipynb)

### `LCEMGP`

context / task を multi-output として扱い、context 特徴量や embedding を通じて出力間関係を学習します。

`SACGP` / `LCEAGP` が集約報酬を扱う contextual model であるのに対し、`LCEMGP` は context 別出力を観測できる場合に向きます。

**Notebook:** [14_contextual_gp.ipynb](../examples/notebooks/14_contextual_gp.ipynb)

## 15. 実務的な選び方

1. まず `SingleTaskGP` をベースラインにする。
2. カテゴリがあるなら `MixedSingleTaskGP`。
3. 明確な fidelity があるなら `SingleTaskMultiFidelityGP`。
4. 関連タスク間で情報共有したいなら `MultiTaskGP` / `KroneckerMultiTaskGP`。
5. 複数目的を独立に学習するなら `ModelListGP`。
6. データ量が大きければ `SingleTaskVariationalGP`。
7. 高次元・少数有効変数なら SAAS / MAP-SAAS。
8. 外れ値なら Student-t / Contaminated / Relevance Pursuit の仮定を比較する。
9. 入力依存ノイズなら Heteroskedastic、反復測定があるなら Replicate Noise を検討する。
10. 入力自体が不確かなら Uncertain Input、関数の滑らかさが場所で変わるなら Nonstationary GP を検討する。
11. 出力がスペクトル・画像・時系列なら structured-output GP.
12. 条件によって変数の有効/無効が変わるなら hierarchical GP。
13. task ごとに特徴量構造が異なるなら `HeterogeneousMTGP`。
14. 明示的な context 構造があるときだけ contextual GP を選ぶ。

モデルの特殊性が高いほど、「使えるから選ぶ」のではなく、そのモデルが仮定するデータ構造が実問題に一致しているかを優先してください。

## 16. 全Notebook一覧

1. [SingleTaskGP](../examples/notebooks/01_single_task_gp.ipynb)
2. [MixedSingleTaskGP](../examples/notebooks/02_mixed_single_task_gp.ipynb)
3. [SingleTaskMultiFidelityGP](../examples/notebooks/03_multi_fidelity_gp.ipynb)
4. [MultiTaskGP / KroneckerMultiTaskGP](../examples/notebooks/04_multitask_gp.ipynb)
5. [ModelListGP](../examples/notebooks/05_model_list_gp.ipynb)
6. [SingleTaskVariationalGP](../examples/notebooks/06_variational_gp.ipynb)
7. [PairwiseGP](../examples/notebooks/07_pairwise_gp.ipynb)
8. [Fully Bayesian SAAS](../examples/notebooks/08_saas_gp.ipynb)
9. [MAP-SAAS / OrthogonalAdditiveGP](../examples/notebooks/09_map_saas_and_additive_gp.ipynb)
10. [RobustRelevancePursuitSingleTaskGP](../examples/notebooks/10_robust_gp.ipynb)
11. [HigherOrderGP / LatentKroneckerGP](../examples/notebooks/11_structured_output_gp.ipynb)
12. [Hierarchical GP](../examples/notebooks/12_hierarchical_gp.ipynb)
13. [HeterogeneousMTGP](../examples/notebooks/13_heterogeneous_multitask_gp.ipynb)
14. [Contextual GP](../examples/notebooks/14_contextual_gp.ipynb)
15. [Student-t / Contaminated observation models](../examples/notebooks/15_robust_observation_models.ipynb)
16. [Heteroskedastic / Replicate Noise](../examples/notebooks/16_noise_models.ipynb)
17. [Uncertain Input / Uncertain Categorical](../examples/notebooks/17_uncertain_input_gp.ipynb)
18. [Nonstationary GP](../examples/notebooks/18_nonstationary_gp.ipynb)
19. [PCA / PLS / Random Projection](../examples/notebooks/19_reduced_gp.ipynb)
20. [Neural Reduction GP](../examples/notebooks/20_neural_reduction_gp.ipynb)
21. [Reduced MultiTask GP](../examples/notebooks/21_reduced_multitask_gp.ipynb)
22. [Mixed Reduced GP](../examples/notebooks/22_mixed_reduced_gp.ipynb)
23. [High-dimensional BO benchmark](../examples/notebooks/23_high_dimensional_bo_benchmark.ipynb)

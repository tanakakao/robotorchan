# 28. Causal Bayesian Optimization：理論とrobotorchan実装の対応

[理論](../theory/28_causal_bayesian_optimization.md) /
[実践Notebook 58](../../examples/notebooks/58_causal_bayesian_optimization.ipynb) /
[比較Notebook 59](../../examples/notebooks/59_causal_bo_benchmark.ipynb) /
[Benchmarkの解釈](../benchmarks/README.md)

本章は、[因果ベイズ最適化の理論](../theory/28_causal_bayesian_optimization.md)で定義した介入と改善量が、実際にどのモジュールで処理されるかを順に追います。使用コード、図、実験ループはNotebookにまとめています。

## 1. 全体像：数式と実装の対応

| 理論の部品 | 実装と所有モジュール | 処理・境界 |
| --- | --- | --- |
| 既知DAG $G$ | `CausalGraph` — `robotorchan.causal.graph` | 親集合、ノード集合、非巡回性、安定したトポロジカル順序 |
| 構造方程式 $f_j$ | `StructuralCausalModel` — `robotorchan.causal.scm` | ユーザー提供の親値から各ノードを順序評価 |
| `do(S=x_S)` | `HardIntervention` — `robotorchan.causal.intervention` | ノードの方程式を固定値で置換 |
| 介入データ $\mathcal D_S$ | `CausalObservation` — `robotorchan.causal.data` | 観測値と実際の介入有無を明示して分離 |
| $f_S$ のGP | `InterventionGPSurrogate` — `robotorchan.causal.surrogate` | 完全に同じ介入集合のデータだけで既存GPを学習 |
| $\arg\max_{S,x_S}\mathrm{EI}/c(S)$ | `propose_causal_intervention` — `robotorchan.causal.proposal` | BoTorchの`LogExpectedImprovement`と既存Optimizerに委譲 |
| 次の実験と履歴 | `CausalBOStudy` / `CausalTrial` — `robotorchan.causal.workflow` | Sobol初期化、`ask`・`tell`、状態と再学習 |
| 実測比較 | `run_causal_benchmark` — `robotorchan.benchmarks.causal` | 同一初期介入の3手法・最良値・コスト・regretを記録 |

公開入口は `robotorchan.causal` にある各クラス・関数です。`robotorchan.benchmarks` にベンチマークの公開APIがあります。実装モジュール同士は集約パッケージではなく定義元モジュールを直接importします。

## 2. DAGとSCMの内部経路

`CausalGraph`のコンストラクタは親ノードの存在、親集合内の重複、自己ループ、循環を検査し、`topological_order`と`parents_of`を提供します。このDAGは**事前にユーザーが与えるもの**であり、観測値から学習・識別する処理はありません。

### 2.1 既知DAGの入力形式

`CausalGraph`は**既知の有向非巡回グラフ**（DAG）を共通表現として保持します。従来の親ノード辞書に加え、Pythonのエッジリスト、隣接行列、NetworkX `DiGraph`から生成できます。どの形式でも同じDAG検証（ノード名・自己ループ・循環）が適用され、`StructuralCausalModel`や`CausalBOStudy`への引数は変更ありません。グラフ構造をデータから推定する機能ではありません。

```python
from robotorchan.causal import CausalGraph

# 従来どおり：ノード -> 親ノードの列
parents = {"X": (), "Z": ("X",), "Y": ("X", "Z")}
graph = CausalGraph(parents)

# 有向エッジリスト：原因 -> 結果
graph = CausalGraph.from_edges([("X", "Z"), ("X", "Y"), ("Z", "Y")])

# 隣接行列：行が原因、列が結果、A(i, j) == 1 が i -> j
nodes = ("X", "Z", "Y")
adjacency = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
graph = CausalGraph.from_adjacency_matrix(adjacency, nodes=nodes)

# NetworkX は必要な場合にだけ使用（pip install robotorchan[causal]）
# import networkx as nx
# graph = CausalGraph.from_networkx(nx.DiGraph([("X", "Z"), ("Z", "Y")]))
```

隣接行列にはNumPy配列・密なPyTorch Tensor・Pythonの二次元列を渡せますが、値は**0または1のみ**です。重み付き行列、無向グラフ、不明方向エッジ、循環は対象外です。行列では`nodes`が必須で、その並びを行と列の共通順序として扱います。エッジリストでは`nodes`を省略できますが、**孤立ノード**を残すには`nodes=("X", "Z", "Y", "isolated")`のように明示してください。すべてのノード名は空でない文字列です。

`graph.to_edges()`、`graph.to_adjacency_matrix()`で変換でき、隣接行列の順序は`graph.nodes`と一致します。NetworkXが必要な場合のみ`graph.to_networkx()`を呼びます。元のグラフと異なる新しい`DiGraph`を返すため、変更しても`CausalGraph`の内容は変わりません。

`StructuralCausalModel.evaluate(intervention)`はトポロジカル順にノードを処理します。各ノードについて、介入値が与えられていればその値を採用し、そうでなければ`equations`辞書から該当ノードの関数を取り出し、親ノードの値を渡して評価します。Hard interventionを実行したときの「**そのノードの生成方程式だけを置換**」という理論操作に対応します。

`CausalObservation`は`values`と`intervention`を一緒に保持します。`intervention=None`は観測のみを意味し、`is_interventional`は`False`です。介入レコードには、固定したノードの値と`values`中の値が完全一致する契約があります。これにより、観測しただけの`X=x`を実行済みの`do(X=x)`として誤登録しにくくしています。

## 3. 介入集合単位のGP

`InterventionGPSurrogate(graph, outcome, intervention_nodes, observations)`は、`record.intervention`が存在し、そのキー集合が指定`intervention_nodes`と**完全一致する**レコードだけを学習に用います。別集合での実験はたとえ共通の`X`を操作していても混ぜません。`intervention_nodes`の順序は`graph.topological_order`に合わせられ、`train_X`は`[n,d_S]`、`train_Y`は`[n,1]`です。少なくとも2件の該当介入記録が必要です。

GPそのものを新しく実装してはいません。`robotorchan.models.standard.single_task.SingleTaskGP`を生成し、`fit()`を委譲します。`posterior(X)`はBoTorch標準の事後オブジェクトを返し、入力末尾の次元が`d_S`であることを検査します。GPの不確実性は学習データとカーネル仮定に依存し、因果識別の保証ではありません。

## 4. LogEI・介入コストとOptimizerの接続

`propose_causal_intervention`は学習済みGPの集合を受け取り、全モデルの`train_Y`から**共通の改善基準`best_f`**を計算します。介入集合ごとに、`LogExpectedImprovement(model, best_f, maximize=...)`を生成して、`robotorchan.optim.dispatch.optimize_acqf(..., q=1)`へ渡します。候補座標はそれぞれの介入集合の`[1,d_S]`です。

各集合の獲得値は元の目的値ではなく**LogEI**です。選択スコアは、集合ごとの固定コスト`c(S)>0`について

$$
\mathrm{score}(S)=\log\mathrm{EI}_S(x_S^\star)-\log c(S)
$$

として比較します。`log_expected_improvement`・`intervention_cost`・`log_utility`を`CausalInterventionProposal`に分けて返すのは、目的関数の実測値と意思決定のスコアを混同しないためです。コストの指定がない集合は1です。

このAPIではすべてのモデルが**同一のDAGインスタンス・同じ目的ノード**を参照し、介入集合が重複してはいけません。これは共通の改善基準で比べるための実装上の前提です。`direction="minimize"`も使用できますが、Batch・Mixed・候補値依存コストは現在のPhaseの対象外です。

### 4.1 ユーザー指定のBoTorch獲得関数による介入候補選択

LogEI以外の獲得関数でも介入集合ごとの候補を提案する場合、`CausalAcquisitionProposal`と`optimize_causal_acqfs`（`src/robotorchan/causal/proposal.py`）を利用します。獲得関数自体の生成は利用者側または上位処理が担当し、たとえばBoTorchの`PosteriorMean`や`UpperConfidenceBound`を既存の`InterventionGPSurrogate.model`から構築できます。独自の獲得関数ファクトリやモデルの再ラップはありません。

**単一集合**では、従来どおりBoTorchの`AcquisitionFunction`を`robotorchan.optim.dispatch.optimize_acqf`へ直接渡すだけで十分です。`optimize_causal_acqfs`は、**複数の介入集合から1つを選ぶ**ための薄いヘルパーです。

このヘルパーでは、キーを`("X", "Z")`のような**順序付きノードtuple**、値を構築済みのBoTorch `AcquisitionFunction`とする辞書を渡します。キーの順序が各獲得関数へ渡される候補テンソル`X[..., 1, d_S]`の列順です。同じ介入集合を並べ替えて重複登録することはできません。各集合は既存の`optimize_acqf(..., q=1)`で独立に最適化します。

集合間で獲得関数値を比較できるとは限らないため、`score(nodes, native_acq_value)`を**必須引数**として明示します。戻り値は有限な実数とし、その最大値を選びます。返す`CausalAcquisitionProposal`の`acquisition_value`はBoTorchの生の獲得値、`ranking_score`はこのスコアです。

`LogExpectedImprovement`なら共通`best_f`を用いた値に対して`log_ei - log(cost)`の補正が使えます。一方、`PosteriorMean`や`UpperConfidenceBound`の値は改善量ではなく、同じコスト式を機械的に適用しません。利用者は比較する獲得関数の目的・尺度・パラメータを揃える必要があります。

従来の`propose_causal_intervention`は互換性のためLogEI専用のままです。全介入集合に共通の`best_f`を設定し、内部で上記の薄い選択処理を利用します。既存の`CausalInterventionProposal`にある`log_expected_improvement`、`intervention_cost`、`log_utility`の意味は変更していません。

### 4.2 GP-SCM：条件付きGPとdo介入の伝播

`GPStructuralCausalModel`（`src/robotorchan/causal/gp_scm.py`）は既知DAGと`CausalObservation`から、**ひとつの構造因果モデル**として各ノードの生成機構を学習します。既存の`StructuralCausalModel`は決定論的方程式の評価であり、GP-SCMとは独立です。従来の`InterventionGPSurrogate`は「介入集合ごとの観測応答GP」であり、条件付き機構を共有するGP-SCMとは学習対象が異なります。

非rootノード`V`には、`CausalGraph.parents_of(V)`で定まる親列を入力とする既存`SingleTaskGP`を作成して`fit()`を委譲します。**V自身が介入された記録だけ**をVの機構学習から除外します。親ノードへの介入ではVの機構を置換しないため、親値とV値が実測されていれば学習に利用できます。観測・介入のどちらであっても、必要な親・子の測定値が欠けたレコードはそのノードの学習から除外します。ノードごとに少なくとも2件の有効なレコードが必要です。

親のないrootノードは、介入されていない実測値から平均と標本標準偏差を推定し、**経験的な正規近似**で扱います。観測値が一定ならそのrootは決定論的とし、不明なrootに架空の標準正規分布を与えません。各非rootのモデルは`model.mechanisms`から取得でき、`model.mechanism_posterior(node, X)`は通常のBoTorch `Posterior`を返します。rootはGPではないため`mechanism_posterior`の対象外です。

推論時は、介入されたノードの親からの流入を切り、指定値で生成機構を置換します。予測対象に必要な祖先だけをDAG順に処理し、その他のノードではroot分布または条件付きGPの`posterior(...).rsample()`で不確実性を伝播します。**`P(Y\mid X=x)`と`P(Y\mid do(X=x))`は異なります**。これらの出力を混同しません。

- `sample_intervention(HardIntervention(...), outcome="Y", sample_shape=torch.Size([1024]))`は単一介入の`[1024]`のTensorサンプルを返します。
- `sample_interventions(X, intervention_nodes=("X",), outcome="Y", ...)`は候補`X[..., q, d_S]`に対して`[*sample_shape, ..., q]`のサンプルを返し、Tensor経路の自動微分を保ちます。ここでの`q`は評価点の次元であり、q-batch最適化を実装するものではありません。
- `observation_noise`を指定すると非root GPの観測ノイズを含む`Posterior`を使います。rootの分散は外生的ばらつきとして常に伝播します。

機構ごとの不確実性を伝播させた介入予測は一般に**ガウス分布ではありません**。GP-SCM全体を単一のBoTorch`Model.posterior()`として公開せず、サンプルをそのまま返します。この段階ではBoTorchの解析的EIやqLogEIへ直接接続できると主張しません。モデル単体の介入サンプリングと、次段階のStudyへの候補選択統合を分離しています。

既知DAGの因果的妥当性、未観測交絡の不在、機構の不変性、介入する親値の十分な学習データ範囲を前提とします。観測データがあるだけで因果効果を自動識別できるという保証ではありません。

## 5. CBO Workflowと試行状態の保持

`CausalBOStudy`は通常の`Study`と異なり、試行の入力を**変数値だけでなく介入集合付き**の`HardIntervention`として管理します。`CausalTrial`は`number`・`intervention`・`state`・`observation`・`proposal`を保持し、状態には共通の`TrialState`を再利用します。

- **Cold start**：各介入集合で`n_initial_trials`件（2件以上）の成功した介入データが揃うまで、既存の`sample_candidate_batches(..., method="sobol")`を使用します。
- **Model phase**：`ask()`が`fit()`で集合別GPを学習し、`propose_causal_intervention`で次の`HardIntervention`を提案します。
- **実験と登録**：`tell(trial, measured_outcome)`または`tell(trial, {"Y": measured_outcome})`で、実測値を介入記録として登録します。`TrialState.FAIL`の場合は学習データを増やしません。
- **更新**：成功`tell()`は学習済みGPのキャッシュを無効化し、次回のモデル段階の`ask()`で再学習します。

`trials`には失敗・実行中の試行も残り、`observations`は成功した介入実験だけを含みます。`q=1`の逐次実験のみを想定するため、実行中のTrialがある状態で次の`ask()`はできません。候補提案が、実際の装置で介入を実施する機能ではない点に注意してください。

## 6. Benchmarkの実装と評価契約

`make_causal_synthetic_scenario`は既知の4ノードDAGと決定的なSCMを返します。`run_causal_benchmark`は同じシードについて各集合2件ずつの**共通初期実験**を作り、CBO、固定`do(X,Z)`の通常GP-BO、Random Searchを比べます。普通のGP-BOはその初期履歴から`do(X,Z)`のデータだけで`SingleTaskGP`を学習し、`LogExpectedImprovement`と`optimize_acqf`で値を最適化します。

`CausalBenchmarkStep`は`outcome_truth`、`best_truth`、`simple_regret`、`evaluation_cost`、`cumulative_cost`を評価ごとに持ちます。`CausalBenchmarkTrajectory`の`best_curve`・`regret_curve`・`cost_curve`を、`robotorchan.benchmarks.comparison.summarize_curves`で複数シード集計できます。

**同じ評価回数と同じ累積コストは異なる条件です。** さらに固定集合GP-BOとCBOでは探索可能な介入集合の種類が異なります。Notebook 59の図だけで「因果推論が通常GPより優れる」と結論づけてはいけません。SCMには介入と反事実に必要な構造を入力しており、構造推定アルゴリズムの比較ではありません。

## 7. 現状の対応範囲

現在の機能は既知DAG、ユーザー指定の決定論的SCM、確率的GP-SCMによる機構学習と介入サンプリング、連続的なHard intervention、単一目的、集合ごとの独立GP、正の固定介入コストです。GP-SCMのサンプリングでは複数候補Tensorの評価ができますが、Studyと候補最適化は引き続き`q=1`です。DAGの因果方向が誤っている場合や、現実の未観測交絡が支配的な場合の因果効果を保証しません。並列試行、観測データのみからの因果識別、構造探索、候補に依存するコストやカテゴリ変数は実装していません。

詳細な実行手順は[Notebook 58](../../examples/notebooks/58_causal_bayesian_optimization.ipynb)、評価プロトコルは[Benchmark資料](../benchmarks/README.md)を参照してください。

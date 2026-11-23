# Causal BO — BoTorch-first 設計契約（Phase 1）

- 対象: [robotorchan](https://github.com/tanakakao/robotorchan)
- 調査基準: `main` @ `5d03d2bf619a378cf1c1c0165a0698e0fdab100b`（2026-11-23）
- 状態: Phase 1 設計確定。**実装・公開API追加はこのPRに含めない**
- 後続: Phase 2（NetworkX）→ Phase 3（獲得関数・単発提案）→ Phase 4（確率的GP-SCM）→ Phase 5（Study統合）→ Phase 6（資料・Benchmark）
- 原則: BoTorch / GPyTorch / PyTorch 公開API → robotorchan既存API → 最小限の新規コード。Monkey patch・private API・不要なaliasを作らない。

## 1. 既存APIの事実確認

| 公開入口 | 主要実装 | 現行の責務 | 互換性 |
| --- | --- | --- | --- |
| `CausalGraph` | `causal/graph.py` | 文字列ノードの親辞書、DAG検証、トポロジカル順序 | 既存コンストラクタ維持 |
| `StructuralCausalModel` | `causal/scm.py` | 利用者指定の決定論的構造方程式を`evaluate` | 変更しない |
| `HardIntervention` | `causal/intervention.py` | 有限実数で指定したhard do介入 | 変更しない |
| `CausalObservation` | `causal/data.py` | 観測／実介入を区別した記録 | 変更しない |
| `InterventionGPSurrogate` | `causal/surrogate.py` | 完全一致する介入集合の実験結果を`SingleTaskGP`で学習 | `fit` / `posterior` / `model`を維持 |
| `CausalInterventionProposal` / `propose_causal_intervention` | `causal/proposal.py` | 複数集合のLogEI/costを比較し`q=1`を返す | 戻り値・従来の既定動作を維持 |
| `CausalBOStudy` / `CausalTrial` | `causal/workflow.py` | Sobol初期化、ask/tell、`TrialState`、GPキャッシュ | 既定経路とTrial契約を維持 |

`robotorchan.causal.__all__`にある9個の名前が現行の公開エントリポイント。上記のすべてはその定義元からもimportできる。CIのwheel smoke testでは`CausalBOStudy`、`CausalGraph`、`HardIntervention`、`InterventionGPSurrogate`、`propose_causal_intervention`を参照している。Phase 2以降も破壊しない。

既存GPは`robotorchan.models.standard.single_task.SingleTaskGP`（BoTorch `SingleTaskGP`を継承した`fit`付モデル）。`InterventionGPSurrogate.model`がnative獲得関数への入口で、`posterior(X)`はBoTorch `Posterior`を返す。既存最適化は`robotorchan.optim.dispatch.optimize_acqf(acq_function, bounds, q, ...)`へ委譲する。`CausalBOStudy`は既存`TrialState`と`sample_candidate_batches(method="sobol")`を再利用している。

現行制限：同じ介入集合の実測データのみをGP学習に使う。集合間の共有なし。複数集合の選択は学習済みモデルを用いた固定`LogExpectedImprovement`、共通`best_f`、`log(EI)-log(cost)`で実施。任意の獲得関数や観測データからの介入推定には未対応。

## 2. 概念とデータの契約

次を混同しない。

| 概念 | 数学的対象 | データ・API上の意味 |
| --- | --- | --- |
| 観測分布 | `P(V)` / `P(Y | X=x)` | `CausalObservation(intervention=None)`。介入の実測値とは異なる |
| 介入分布 | `P(Y | do(S=x_S), D, G)` | ノードの生成機構を置換して伝播させた予測。観測条件付き分布の単純な別名ではない |
| 介入応答関数 | `m_S(x_S)=E[Y | do(S=x_S)]` | 介入集合別GPが直接近似する対象 |
| GP-SCM | `V_j=f_j(Pa_j,U_j)`の条件付き機構をGP等で表現 | **1個の因果モデル**の内部に複数の条件付きモデルを所有 |
| 獲得関数 | `a_S(x_S; D)` | 介入に関するモデル予測から候補値を評価。集合間比較には共通の目的尺度・基準が必要 |
| 最適化 | `argmax_{S,x_S} score_S(x_S)` | `optimize_acqf`が各集合の連続値を最適化。集合間のrankのみcausal層 |

単一介入集合の学習・予測はBoTorchそのものに近い形で利用できる。`CausalBOStudy`は必須ではない。

```python
surrogate = InterventionGPSurrogate(graph, "Y", ("X",), records)
surrogate.fit()
posterior = surrogate.posterior(test_X)
acqf = LogExpectedImprovement(model=surrogate.model, best_f=best_f)
candidate, value = optimize_acqf(acqf, bounds_tensor, q=1)
```

注：`best_f`と`bounds_tensor`は利用者がその集合と目的尺度に合わせて指定する。`model=surrogate`ではなく`model=surrogate.model`を渡す（現行wrapper自体はBoTorch `Model`のサブクラスではない）。独自のモデル互換ラッパーは不要。

## 3. Phase 2 — グラフの相互運用（実装）

Phase 2はユーザー要望に合わせてNetworkX専用から**複数の入力形式**へ変更した。`CausalGraph`の辞書コンストラクタはそのまま残し、以下の**独立した明示的コンバータ**を追加する。`from_any`のような曖昧な自動判定や、新しいDAGフレームワークは作らない。

| 入力 | API | 向き・順序 |
| --- | --- | --- |
| 親ノード辞書（従来） | `CausalGraph({"X": (), "Y": ("X",)})` | `node -> parents` |
| 有向エッジリスト | `CausalGraph.from_edges([("X", "Y")], nodes=("X", "Y"))` | `(source, target)` は `source -> target` |
| 隣接行列（list / NumPy / dense torch） | `CausalGraph.from_adjacency_matrix(A, nodes=("X", "Y"))` | `A(i, j)=1` は `nodes[i] -> nodes[j]` |
| NetworkX DiGraph | `CausalGraph.from_networkx(nx_graph)` | `networkx.DiGraph` の有向エッジ |

- 出力は`to_edges()`、`to_adjacency_matrix()`、`to_networkx()`。行列は**0/1のみ**、順序は`CausalGraph.nodes`。Node名は従来どおり非空文字列（数値から文字列への自動変換はしない）。
- 孤立ノードはエッジリストで`nodes=`を指定するか、辞書・行列・NetworkXの明示ノードから保持する。エッジリストに`nodes=`を与えない場合は初出順に推定する。
- `networkx`は`pip install robotorchan[causal]`で使えるoptional dependency。使用時だけimport。無向・多重・循環・非対応ノード型は明示エラー。
- 従来の`CausalGraph`のDAG検証へ集約し、`StructuralCausalModel`、`CausalBOStudy`の既存引数を変更しない。
- pgmpy・causal-learnの専用アダプタは追加しない。公開NetworkX変換が利用可能な場合は利用者側で変換する。
- Phase 2の範囲は型変換とガイドおよび対象テストに限定する。GP-SCM・最適化の実装はPhase 3以降とする。

## 4. Phase 3 — native獲得関数と複数介入集合の責務分離（実装）

### 単一集合：追加APIなし

BoTorch `AcquisitionFunction`インスタンスを`optimize_acqf`に**直接**渡す。獲得関数ファクトリ、文字列からのディスパッチ、モデルの再ラップを新設しない。

### 複数集合：必要な時だけ薄い束ね役

`propose_causal_intervention(surrogates, bounds, ...)`はLogEI専用のまま公開互換性を維持。内部の最適化・候補検証・順位付けを再利用するため、以下の**最小限の汎用補助API**を実装した：

```python
optimize_causal_acqfs(
    acqfs: Mapping[tuple[str, ...], AcquisitionFunction],
    bounds: Mapping[str, tuple[float, float]],
    *,
    score: Callable[[tuple[str, ...], float], float],
    optimizer: OptimizerName = "botorch",
    num_restarts: int = 5,
    raw_samples: int = 64,
) -> CausalAcquisitionProposal
```

- キーは**順序付き介入ノードtuple**で、BoTorch `X[...,q,d_S]`の列順を表す。`frozenset`による無順序な入力座標の取り違えを避ける。異なる順序で同一集合が2回出た場合は拒否。
- 値は**利用者が構築済みの獲得関数インスタンス**。`acquisition_factory`は導入しない。各集合で通常の`optimize_acqf(..., q=1)`を呼ぶ。
- `score(nodes, native_acq_value)`は比較尺度を**明示的に**与える関数。任意の獲得値を無条件に比較可能とはみなさない。入力は最適化後のスカラー値、戻り値は有限実数。負の値やログ値のcost処理を勝手に推定しない。
- `CausalAcquisitionProposal`は`intervention: HardIntervention`、`acquisition_value: float`、`ranking_score: float`のみを持つ最小の新しい結果型。既存`CausalInterventionProposal`の`log_expected_improvement`/`log_utility`という**LogEI固有のフィールドを別の獲得関数に流用しない**。
- `score`は既存LogEI経路では`log_ei - log(cost)`。一般の`UCB`や`PosteriorMean`は改善量でないため、この変換を適用しない。
- 複数集合で比較可能な実務上の条件：同じターゲット、同じ目的の単位・方向、同じ改善基準（EI系）、同じ獲得関数の定義・ハイパーパラメータ。異なるモデル出力変換やコスト正規化に注意する。
- `propose_causal_intervention`の旧来の戻り値を維持し、候補選択のみ薄い補助処理へ委譲する。nativeの`acqf`を専用ラッパーで包まない。

Phase 3の対象テストは`tests/causal/test_native_acquisition_selection.py`および従来の`tests/causal/test_phase4_proposal.py`。native`PosteriorMean`、`UpperConfidenceBound`、`LogExpectedImprovement`、複数集合の次元・順序・明示コストスコア、不正な候補／スコア、従来の最小化・LogEIの扱いを確認する。大規模なOptimizer互換性監査は行わない。

## 5. Phase 4 — 確率的GP-SCMの実装契約

### 既存SCMとは別クラス

`GPStructuralCausalModel`を独立した公開クラスとして実装。決定論的な`StructuralCausalModel.evaluate`を変更せず、**1つの確率的な構造因果モデル**を構成する。

```python
model = GPStructuralCausalModel(graph, observations)
model.fit()
samples = model.sample_intervention(
    HardIntervention({"X": 0.4}),
    outcome="Y",
    sample_shape=torch.Size([1024]),
)
mean = samples.mean(0)
variance = samples.var(0, unbiased=False)
```

上記は実装したモデル単体APIの例。`sample_intervention`はモデルが返す観測結果のサンプルであり、BoTorch `Posterior`であると偽らない。summaryは既存Tensorの集約操作を利用し、不必要な独自posterior型を公開しない。モデルから介入に関係なく`fit`、ノード条件付きモデルの予測、do予測を独立して使える設計とする。

実装ファイル：`src/robotorchan/causal/gp_scm.py`。テスト：`tests/causal/test_gp_scm.py`。入力の観測・介入状態は既存`CausalObservation`と`HardIntervention`をそのまま使用。新たな独自GP、事後分布クラス、専用SCM用Optimizerは追加しない。

### 学習規則

- 入力：既知DAGと`CausalObservation`。観測データと介入データを区別して受け取る。
- 非rootノード`V_j`：実際に**そのノード自体へ介入していない**記録のみから`P(V_j | Pa_j)`を既存`SingleTaskGP`等で学習。親の値とノード値が揃ったレコードに限定する。親への介入があっても`V_j`自身の生成機構が変わっていなければ利用可能（mechanism invarianceが前提）。
- 親なしノード：入力次元0の`SingleTaskGP`を無理に生成せず、観測データからrootの分布を表す**既存PyTorchの分布**で扱う（決定論的rootや近似上の退化分布も検討）。未観測rootを無条件に標準正規と仮定しない。
- `do(S=x_S)`では介入ノードの条件付き機構をスキップして値を固定し、他のノードだけをトポロジカル順に伝播する。対象の予測には条件付きGPの不確実性と必要な観測ノイズを区別して伝播させる。
- 各ノードの親列は`CausalGraph.parents_of`に従い、dtype/device、未fit、親値欠測、学習に必要なレコード不足を明示エラーにする。
- 再学習は既存GPの`fit`を使用し、成功した観測追加時に必要なノードモデルを再構築する。新しいGP、カーネル、MLLは実装しない。
- 介入集合間の情報共有は**共通する条件付き機構から生じる**。介入集合間の同一データの単純な混合ではない。

### 推論の前提・識別可能性

初期対応は、因果的に正しい既知DAG、関連変数の観測、未観測交絡がない（因果十分性）、介入で他の機構が不変、対応する親の領域にデータの支持があることを基本前提とする。欠測した親・ノードや支持範囲外への外挿について推定品質を保証しない。未識別の場合には適切な例外または明示的な非対応を返す。観測データを何でも`do`分布に変換できると主張しない。

予測対象は以下を区別する。

1. `P(V_j | Pa_j,D)`：条件付き機構のGP posterior
2. `P(Y | do(S=x_S),D,G)`：構造に沿って伝播した介入予測分布
3. `m_S(x_S)=E[Y | do(S=x_S),D,G]`：最適化の目的となる介入応答平均

不確実性伝播を伴う非ガウスな介入予測を、解析的ガウスEIへ無条件に渡してはならない。

### BoTorchへの接続

- **実装済み**：モデル単体の`fit`、`sample_intervention`、Tensorベースの`sample_interventions(X, intervention_nodes, outcome)`、`mechanism_posterior`。BoTorchの候補Tensor `X[..., q, d_S]`を直接受けて微分を伝播し、`HardIntervention`（Python floatへ変換）を微分経路に使用しない。
- **条件付き**：標準`Posterior`の`rsample`、`base_sample_shape`、`rsample_from_base_samples`、dtype/device、batch/q/event形状、必要な自動微分の契約を満たせる場合に限り、BoTorchの既存MC獲得関数（例：qLogEI系）へ接続する。完全な契約を満たす前に`botorch.models.Model`の継承や`posterior()`互換性を宣言しない。
- **現段階**：構造に沿ったMCサンプル評価のみ公開し、BoTorch `Model`や`Posterior`の完全互換を主張しない。非rootの条件付きGPには従来の`SingleTaskGP.posterior`を再利用する。直接的なMC獲得関数・Study統合は後続Phaseの担当。
- 候補値に対する推論でMC乱数が変わると獲得関数の数値最適化が不安定になるため、共通乱数・固定base samples等の扱いをPhase 4〜5の狭い機能テストで確認する。

## 6. Phase 5 — Study統合契約（実装）

- `CausalBOStudy`の既定値と既存`ask()`/`tell()`契約は**介入集合別GP方式**のまま維持する。
- `model_type="gp_scm"`を明示したときのみ`GPStructuralCausalModel`を`fit()`してキャッシュする。初期`observational_data`は純観測`CausalObservation(intervention=None)`限定で、Studyの成功Trialとは別管理。
- `propose_gp_scm_intervention(..., best_f, ...)`を公開し、Studyなしでも単発で介入集合と値を最適化可能。内部`_GPScmMCLogEI`はBoTorch `AcquisitionFunction`を利用するが、構造モデルを偽の`Model`/`Posterior`に変換しない。
- MCの各評価では固定seedとfork_rngを使い再現性を確保する。形状の違う評価間の完全な共通base samplesは未保証であり、MC EIの0改善領域では勾配が平坦になり得る。改善基準`best_f`は実介入の成功Trialのみで決める。

- 新しいStudyループを作らず、`CausalBOStudy`の既存のTrial状態・履歴・`ask()`/`tell()`を再利用し、モデル構築・提案経路だけを切り替える。
- `tell(trial, scalar)`は従来互換で許容するが、**中間ノード`Z`等を測定しない限り**GP-SCMの条件付き機構に十分なレコードが確保できず、`fit()`時に明示的な不足エラーとなる。観測していないノードをSCMから勝手に「実測」として補完しない。
- 完了／失敗状態、単一RUNNING試行、Trial番号、初期Sobol、履歴、コストは既存処理を再利用。GP-SCM方式は介入による実測`best_f`を確保する**最初の1試行のみSobol**とし、その後は条件付き機構の学習データ要件に従う。集合ごとの`n_initial_trials`は要求しない。
- モデル単体のdo予測とStudyを分離し、`CausalBOStudy`なしの単発提案も公開する。対象テスト：`tests/causal/test_gp_scm_study.py`。

## 6.1. 詳細理論・Notebook・Benchmarkの整合性（Phase 6）

このPhaseでは**GP-SCMの内部GPや獲得関数を再実装しない**。Phase 4/5の既存APIを教材・比較に接続する。

- [理論28章](../../docs/theory/28_causal_bayesian_optimization.md)：truncated factorization、親条件付きGP、root分布、GP-SCMの介入サンプリング、MC-EI、観測と実介入のbest_f区別、識別可能性と数値的な制限を初心者にも追える数式とともに説明
- [Guide 28章](../../docs/guide/28_causal_bayesian_optimization.md)：入力形状、データ行の選別、モデル別fit/posterior、単発提案、Studyによる再学習、MCの限界を公開APIベースで説明
- [Notebook 58](../../examples/notebooks/58_causal_bayesian_optimization.ipynb)：従来集合別GPを残し、GP-SCMの条件付きGP、do分布の分位点、MC-EI、単発候補提案、Ask-and-Tell再学習を追加。グラフあり
- [Notebook 59](../../examples/notebooks/59_causal_bo_benchmark.ipynb)：既存3方式の実測比較にGP-SCMを任意追加し、同一初期介入、cost軸・regret軸を明示
- [Benchmark](../../src/robotorchan/benchmarks/causal.py)：既定の3手法`METHODS`は変更せず、`methods`指定で`gp_scm`を任意追加。全方式同じ初期**実介入**データ、GP-SCMだけへの追加観測はなし
- [CLI](../../benchmarks/causal_bo.py)：`--include-gp-scm`、`--mc-samples`、`--mc-seed`を指定でき、実行条件をsummary JSONに保存
- [対象テスト](../../tests/benchmarks/test_causal_bo.py)：既定方式の互換性、共通初期履歴と追加方式、MC引数の検証

比較する介入集合は同一でも、機構別GP-SCMは中間変数を使用し、集合別GPは目的値を直接学習するという**情報利用法の違い**がある。数値的な優劣を文書やNotebookへ固定値として書かない。未知交絡・DAGの誤指定、観測欠測、rootモデルの近似、不確実性の較正は今回の合成実行だけでは検証できない。

## 7. 後続Phaseの受け入れ条件（境界を固定する）

| Phase | 追加範囲 | 必要な確認 |
| --- | --- | --- |
| 2 | 辞書・エッジリスト・隣接行列・NetworkXのDAG入出力 | 孤立ノード、順序、行列方向、循環、非文字列、optional依存、既存辞書コンストラクタ |
| 3 | native acqfでの集合最適化と比較 | q=1、`LogEI`既存整合、異なる介入次元・順序、costの尺度、数値の有限性 |
| 4 | 条件付きGP-SCMとdoサンプリング | 観測 vs 実介入、介入ノード機構の置換、root処理、欠測、既知SCM、MCの予測特性 |
| 5 | 同じStudyで2方式の選択 | ask/tell、失敗、キャッシュ無効化、観測ノード不足、既存既定挙動、単発下層API |
| 6 | Notebook 58/59、theory 28、guide 28、既存benchmark | 図・候補提案・理論と実装の一致、複数seed、simple regret、コスト、CI |

Phase 6の評価では、データ資源が違う方式を**同じ条件の比較と誤表現しない**。一例として共通観測データと共通介入予算を明示する。後続PRの対象外は因果探索、未観測交絡の一般識別、Categorical/Mixed全面対応、q>1の新規アルゴリズム、ライブラリ横断監査。

## 8. 設計判断（Phase 1時点）

1. 決定論的`StructuralCausalModel`と確率的`GPStructuralCausalModel`は**別の責務**。
2. GP-SCM全体は1モデルだが、条件付きGPは複数存在しうる。ユーザーが「1モデル」を利用することと、内部で1個のGPしか学習しないことは同義ではない。
3. `InterventionGPSurrogate`はすでに十分BoTorch-native。`model`公開を維持し、不要な全面的な`Model`継承改修は行わない。
4. 単一集合で`acqf`オブジェクトを直接渡せるのは現状の強み。汎用ファクトリを増やさない。
5. 複数集合はモデル・候補次元が異なるため、**各集合の最適化結果を比較する薄い処理だけ**がcausal層の責任。
6. 既存のLogEI専用結果型の意味は変えない。
7. GP-SCMの条件付き分布と介入分布を型・ドキュメント双方で区別する。BoTorch MC獲得関数の適合性は実際の`Posterior`契約で判定する。
8. Phase 1に実装変更や網羅的監査は含めない。Phase 2以降に各PRの範囲内で最小限の検証を行う。

### 主なコード根拠

- [causal public exports](../../src/robotorchan/causal/__init__.py)
- [graph](../../src/robotorchan/causal/graph.py) / [deterministic SCM](../../src/robotorchan/causal/scm.py)
- [intervention GP](../../src/robotorchan/causal/surrogate.py)
- [proposal / LogEI](../../src/robotorchan/causal/proposal.py)
- [ask/tell workflow](../../src/robotorchan/causal/workflow.py)
- [existing optimizer dispatch](../../src/robotorchan/optim/dispatch.py)
- [existing SingleTaskGP](../../src/robotorchan/models/standard/single_task.py)
- [existing theory](../../docs/theory/28_causal_bayesian_optimization.md) / [guide](../../docs/guide/28_causal_bayesian_optimization.md)

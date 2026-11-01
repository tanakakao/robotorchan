# Model Selection：問題設定から実装を選ぶ

## 1. 選択の出発点

モデル名より先に観測形式・入力構造・推論方式・意思決定目的を決める。[理論第13章](../theory/13_model_selection.md)と[利用者向けモデル選択](../getting_started/model_selection.md)が基礎となる。通常の連続scalar回帰は `SingleTaskGP` を比較基準にするが、比較観測・分類・構造化出力を無理に通常回帰へ変換しない。

次の順に確認する：観測はnumericかpairwiseかbinaryか／outputはscalarかvectorかtensorか／categorical・hierarchy・contextがあるか／taskとfidelityを区別できるか／異分散・外れ値・入力不確実性のどれが問題か／大きいのは入力次元dか標本数nか／獲得関数が必要とするposterior演算は何か。

## 2. モデルの数学的仮定と公開API

| 問題設定 | 主なモデル | 実装・推論で重要な違い |
| --- | --- | --- |
| 通常の連続回帰 | `SingleTaskGP` | BoTorch Exact GPを継承、robotorchanはraw data保持・fitを拡張 |
| Mixed入力 | `MixedSingleTaskGP` | categorical kernelと連続kernelを区別。カテゴリの数字は通常距離ではない |
| Fidelity構造 | `SingleTaskMultiFidelityGP` | fidelity featureとtarget fidelity、評価費用を分離 |
| 関連task | `MultiTaskGP` / `KroneckerMultiTaskGP` | long formatとblock designは別の観測契約 |
| 独立複数output | `ModelListGP` | 出力間の統計的共分散が必要か検討 |
| 比較観測 | `PairwiseGP` | Gaussian numeric likelihoodではない |
| 構造化tensor | `HigherOrderGP` | posteriorのtensor軸とsamplingを確認 |
| 大標本数n | `SingleTaskVariationalGP` | inducing pointsとVariationalELBO |
| 高次元・軸sparsity | SAAS / MAP-SAAS family | Fully Bayesian NUTSとMAP推定は異なる |
| 入力低次元表現 | PCA / PLS / Neural reduced GP | frozen reducerとjoint encoderの学習契約が異なる |
| 位置依存noise | Heteroskedastic / Joint heteroskedastic GP | latent不確実性とobservation noiseを区別 |
| 外れ値 | Student-t / Relevance Pursuit | heavy-tailedとsparse outlierは異なる生成機構 |
| 二値分類 | `BinarySingleTaskGPClassifier` | latent posterior ≠ class probability |
| Non-GP tabular | Random Forest / Boosting | ensemble disagreementはGP posteriorではなく入力微分にも制約 |

各公開モデルのconstructorや未対応事項は[モデルガイド](../models/README.md)と[旧モデルMapping](../../.github/project-records/theory-guide/mapping/models.md)で確認する。表中の掲載は特定のモデル・獲得関数・optimizerの組合せが検証済みという意味ではない。

## 3. Model → Acquisition → Optimizerの接続

標準の単目的BOは `SingleTaskGP` のposteriorからBoTorch `LogExpectedImprovement`などのutilityを構成し、`robotorchan.optim.optimize_acqf`またはBoTorch optimizerで候補を生成する。`best_f`は最大化する目的の尺度を使う。[獲得関数のImprovement](acquisition/02_improvement.md)と[Gradient optimizer](optimization/02_gradient_based.md)に接続する。

観測ノイズがある場合はX_baselineのlatent uncertaintyを扱うqLogNEI系、複数目的ではobjective尺度・reference point・Pareto partitioningを定義するEHVI/NEHVI系を検討する。複数outputがあるだけで多目的とは限らず、outcome constraintsと既知candidate constraintsも層が違う。[Batch/Noisy](acquisition/04_batch_noisy.md)、[多目的](acquisition/07_multiobjective.md)、[制約](acquisition/08_constraints.md)を参照する。

回帰+分類の意思決定では分類 `posterior(X)`のlatent値を可行確率と混同せず `predict_proba(X)`を使う。`ProblemSemantics`とacquisition compositionの責務は[Classificationガイド](24_classification.md)と[End-to-Endガイド](walkthrough/README.md)で確認する。

高次元ではモデル側のSAASやReducedGPと、探索側のTuRBO / REMBO / HeSBO / ALEBO / BAxUSを別々に選ぶ。[高次元モデル](08_high_dimensional_gp.md)と[高次元探索](20_high_dimensional_search.md)は異なる座標系を扱う。入力次元dが大きい問題と標本数nが大きい問題は同じではない。

## 4. 実行例から辿る

[統合APIスクリプト](../../examples/unified_api_workflow.py)では二出力Exact GPのfit、BoTorch sampler/objective、qSimpleRegret、Sobol candidateをつなぐ。[End-to-End Notebook](../../examples/notebooks/36_end_to_end_workflows.ipynb)と[横断ウォークスルー](walkthrough/README.md)を併読する。前者のスクリプトはcandidate proposalまでであり、真の評価を追加する閉ループ全体とは区別する。

## 5. 検証の境界

モデルを作れることと、posterior sample、fantasize、q-batch、one-shot、mixed constraints、dtype/deviceまで互換であることは別。実装の分類はImplemented / Extended / Delegated / Approximated / Unsupported / Unverifiedを使い分ける。[検証索引](../../docs-internal/verification/theory-guide/README.md)では数式の独立数値検証、契約テスト、CI成功を分ける。旧Mappingは移行台帳の原文区間照合が済むまで保全する。

## 6. モデルを横断した学習・posterior・獲得関数の契約

モデルの選択では「posterior()が存在するか」だけでなく、**その確率変数の意味、学習目的、BoTorch acquisitionへの接続**を順番に調べる。以下は各実装章とソースの対応をたどるための比較表であり、実行済み互換性の網羅表ではない。

| Family / ガイド | 学習目的・推論 | posteriorまたは予測対象 | acquisitionでの注意 |
| --- | --- | --- | --- |
| [Exact GP](02_gaussian_process.md) | `ExactMarginalLogLikelihood` | latent Gaussian予測（observation_noise指定に注意） | analytic EIはGaussian/単一目的等の条件が必要 |
| [Variational GP](09_variational_gp.md) | `VariationalELBO`、inducing variables | 近似latent GP posterior | `num_data`は全観測数。ELBO構築だけではfitしない |
| [Preference](10_preference_learning.md) | `PairwiseLaplaceMarginalLogLikelihood` | pairwise観測からのlatent utility | comparisonsは数値train_Yではなくindexの対 |
| [Higher-Order](11_structured_output.md) | Exact GPのMLLとtensor covariance | tensor-valued response | scalar utilityを定義せずflattenしない |
| [Heteroskedastic](15_heteroskedastic_noise.md) | response/noise交互推定またはjoint `training_loss()` | response posteriorとlog-noise posterior | jointモデルに通常の`make_mll()`を適用しない |
| [Uncertain Input](16_uncertain_input_gp.md) | Gaussian入力不確実性をkernelに積分したExact GP | 元入力のdeterministic候補でのposterior | 訓練入力の誤差と候補実行時の摂動を分ける |
| [Nonstationary](17_nonstationary_gp.md) | Gibbs kernelを持つExact GP | 入力依存lengthscaleのlatent posterior | 局所lengthscaleと観測ノイズを混同しない |
| [DeepGP](22_expressive_gp.md) | DeepApproximateMLL(VariationalELBO) | MC `DeepGPPosterior` | Gaussian analytic acquisitionへの無条件接続を避ける |
| [NNGP / Spectral Mixture](22_expressive_gp.md) | 専用kernelを持つExact GP | Gaussian GP posterior | kernel priorの意味と入力scaling/初期化を確認 |
| [Non-GP](23_non_gp_surrogates.md) | estimator固有`fit()` | empirical `EnsemblePosterior`またはNGBoost予測分布 | 入力勾配・点間依存・校正がGPと異なる |
| [Binary Classification](24_classification.md) | Bernoulli likelihoodのvariational学習 | `posterior(X)`はlatent、`predict_proba(X)`はclass probability | feasibilityへは確率を渡しlatent値を誤用しない |
| [Fully Bayesian SAAS](08_high_dimensional_gp.md) | BoTorch NUTS/MCMC、`supports_mll=False` | hyperparameter uncertaintyを反映するposterior | モデル/獲得関数のMCMC軸とsampling互換性を確認 |

### 6.1 実装を確認する順番

1. **入力形式：** 元入力Xのshapeと、task/fidelity/cat_dims等の構造列を区別する。
2. **学習：** `fit()`、`make_mll()`、`training_loss()`、BoTorch NUTSなどから実際の経路を選ぶ。全modelが同じmethodを実装するとは限らない。
3. **予測：** latent値、観測ノイズ付き予測、class probability、tensor出力、ensemble memberを区別する。
4. **意思決定：** analytic/MC acquisition、classification PoF、gradient-free optimizerの必要条件を確認する。
5. **検証：** source contractの存在、APIを呼べた事実、数式の数値一致、E2E最適化性能を区分する。

### 6.2 単一の多出力モデルと異種モデルcontainer

[MultiTask](07_multitask_multioutput.md)はタスク共分散を扱うモデル、`ModelListGP`は子GPの独立な合成、`HeterogeneousModel`は回帰と分類の出力意味論を保つcontainerである。`HeterogeneousModel`自身が全出力に対する共通Gaussian posteriorを作るわけではない。複数目的を構成する場合は[多目的獲得関数](acquisition/07_multiobjective.md)、分類制約は[制約付き獲得関数](acquisition/08_constraints.md)の共有posterior・出力所有権の制限を参照する。

この比較は章間の意味論整合を支援する読書ガイドであり、全24章の関数・テスト・数値propertyの移行検収を完了とするものではない。

## 7. 学習経路から各章を選ぶ

同じ「GP」でも、学習目的関数と予測対象によって実装経路が変わる。
以下の入口を使い分け、クラス名やモデル選択表だけで互換性を判断しない。

| まず確認したいこと | 読み進める章 | 実装での判断点 |
| --- | --- | --- |
| 通常の回帰を学習・予測したい | [Exact GP](02_gaussian_process.md) | `make_mll()`からExact MLLを組み立てる |
| 多数の観測点を近似的に扱いたい | [Variational GP](09_variational_gp.md) | ELBOの`num_data`とinducingの設定 |
| 観測が二者比較である | [Preference](10_preference_learning.md) | comparisonsとPairwise Laplace MLL |
| ノイズ分散が入力依存する | [Heteroskedastic](15_heteroskedastic_noise.md) | responseとnoiseを分け、jointなら`training_loss()` |
| 高次元で軸方向の疎性を仮定する | [SAAS](08_high_dimensional_gp.md) | MCMCとMAP-SAASを分ける。追加依存を確認 |
| 応答が合否などの二値である | [Classification](24_classification.md) | latent posteriorとclass probabilityを分ける |
| 複数のモデル出力を意思決定へつなぐ | [Multi-Output](07_multitask_multioutput.md) | task共分散と独立出力・異種出力を区別する |

最初の動作確認には[BOの基本例](01_bayesian_optimization.md)から始め、
[獲得関数](04_acquisition_function.md)で候補生成まで追う。
選択表は適用可能性を確認するための案内であり、性能や互換性の保証ではない。

## 8. 入力・出力shapeと構造列を確認する

| モデル | 入力・観測の典型的な形 | 選択時に保つ意味 |
| --- | --- | --- |
| `SingleTaskGP` | X: `[n,d]`, Y: `[n,1]` | 通常の連続値回帰 |
| `MixedSingleTaskGP` | X: `[n,d]`, Y: `[n,1]`, `cat_dims` | 元入力のカテゴリ列を指定 |
| `MultiTaskGP` | X: `[n,d+1]`, Y: `[n,1]` | task列を含むlong format |
| `KroneckerMultiTaskGP` | X: `[n,d]`, Y: `[n,m]` | 全taskが同じXを持つblock design |
| `ModelListGP` | 各子モデル固有のX・Y | 出力間を独立にモデル化 |
| `HeterogeneousMTGP` | taskごとの`train_Xs`など | task固有の特徴空間を明示 |
| `HigherOrderGP` | X: `[n,d]`, Y: `[n,r1,...,rk]` | tensorの各出力軸 |
| `LatentKroneckerGP` | X: `[n,d]`, T: `[m,1]`, Y: `[n,m]` | 波長・時刻などの明示的な出力座標 |
| `PairwiseGP` | datapoints: `[n,d]`, comparisons: `[k,2]` | 勝者・敗者のindex対 |

batch次元や変換の詳細は各モデルガイドで確認する。
少数の独立応答なら`ModelListGP`が候補になり、タスク相関が重要なら
MultiTask系を比較する。例えば強度と導電率の同時最適化でも、
独立GPからqEHVIなどの多目的獲得関数を構成できる。
Multi-objectiveという問題設定だけでは、task共分散が必要かどうかは決まらない。

taskに存在しない特徴を単に0埋めすると、存在しない値に物理的な意味を与える恐れがある。
異種特徴空間には`HeterogeneousMTGP`、親変数で有効次元が変わる場合は
`HierarchicalConditionalKernelGP`や対応するMultiTask版を検討する。
contextごとの加法構造と集約応答には`SACGP` / `LCEAGP`、
context別出力には`LCEMGP`が候補となる。

## 9. 高次元・ノイズ・計算予算を別々に判断する

| 主な問題 | 候補・設定 | 混同しないもの |
| --- | --- | --- |
| 軸方向の有効変数が少ない | Fully Bayesian SAAS / MAP-SAAS | MCMC周辺化とMAP推定 |
| 加法的な応答構造 | `OrthogonalAdditiveGP` | SAASの疎性prior |
| 低次元表現が妥当 | PCA / PLS / Random Projection / AE / VAE | 元空間の候補探索と表現削減 |
| 観測数nが大きい | `SingleTaskVariationalGP` | 入力次元dの大きさ |
| 既知の固定観測分散 | `train_Yvar` / likelihood | 未知の入力依存ノイズ |
| 入力依存ノイズ | heteroskedastic系 | 潜在関数の非定常性 |
| 反復測定から分散を推定 | replicate-noise系 | 外れ値への頑健性 |
| 外れ値・heavy tail | Student-t / contamination / relevance pursuit | 入力誤差・ノイズ分散・regimeの違い |
| 入力値の不確かさ | uncertain-input系 | 候補実行時の摂動 |
| 場所による滑らかさの変化 | nonstationary系 | observation noise |
| 条件やregimeによる変化 | task / categorical / hierarchical系 | 連続的なlengthscale変化 |

Fully Bayesianの`SaasFullyBayesianSingleTaskGP`と
`SaasFullyBayesianMultiTaskGP`のうち、MultiTask版はtask列を持つ形式を使う。
疎な外れ値には`RobustRelevancePursuitSingleTaskGP`などの観測側モデルを検討し、
SAASの入力側の疎性と区別する。MAP-SAASには
`AdditiveMapSaasSingleTaskGP`と`EnsembleMapSaasSingleTaskGP`があり、
NUTSを用いるFully Bayesian SAASとは計算・posteriorの契約が異なる。
Mixed版の有無とカテゴリ変換の所有権もモデルごとに確認する。

通常のExact GPでも観測数やtask数が増えれば重くなるため、
クラス名だけから計算コストの厳密な順位を付けない。
まず小規模な`SingleTaskGP`を基準にし、推論の反復数、MCMC、
reducer / encoder学習、posterior sampling、候補最適化を含めて予算を決める。

## 10. Registryと実験によるモデル選択

[Capability-aware workflow](../capability_workflow.md)は`ProblemSpec`から
互換候補と不適合理由を整理する。Registryへの登録は、
そのモデルが特定のデータで最良となるという性能順位ではない。
「categorical列がある」「高次元である」といった条件は候補を絞る情報であり、
統計的な仮定やposteriorの較正を自動的に検証するものではない。

```text
目的と最適化方向を定義
  → 入出力schema・task・fidelity・context・hierarchyを整理
  → 必要最小限のsurrogateを選ぶ
  → baselineと比較してposteriorを診断
  → acquisitionと候補optimizerを選ぶ
  → 候補のshape・bounds・制約を確認
  → 実験・観測追加・再学習を反復
```

数値スコア、二者比較、絶対的な二値ラベルは観測契約が異なる。
正確な実数応答を取得できるなら回帰GP、人の総合判断を比較で取得するなら
`PairwiseGP`、合否なら分類モデルを比較基準にする。
特殊モデルの採用後も、通常GP、独立出力とtask共有、獲得関数の違いなどを
同じ評価予算・観測条件で比較し、モデルと探索法の効果を分けて評価する。

具体的な学習APIは[学習契約](../development/training_api.md)、
モデルと獲得関数の対応は[モデル選択](../getting_started/model_selection.md)、
探索側の選択は[最適化Guide](optimization/12_selection_guide.md)へ進む。

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

## 実装補足：理論本文から分離した対応情報

### 13.41 robotorchan で推奨する実務フロー

モデル選択から BO 実行までをまとめると次の流れです。

```text
1. optimization target を定義
        ↓
2. input / output schema を整理
        ↓
3. task / fidelity / context / hierarchy を整理
        ↓
4. 最小構造の surrogate を選ぶ
        ↓
5. baseline model と比較
        ↓
6. posterior diagnostics
        ↓
7. acquisition function を選ぶ
        ↓
8. candidate optimizer を選ぶ
        ↓
9. candidate validity を確認
        ↓
10. sequential benchmark / real experiment
```

この順序で考えると、モデルと探索ロジックが混ざりにくくなります。

---

## 理論章から分離した実装上の条件

#### 理論との対応：実装上の補足

`robotorchan` には標準 GP、Mixed、Multi-Fidelity、Multi-task、Variational、SAAS、Preference、Structured
Output、Hierarchical、Contextual など多くのモデルがあります。

#### 13.2 最重要フローチャート：実装上の補足

robotorchan のモデル選択を大きくまとめると次のようになります。

#### Mixed variable：実装上の補足


が候補です。ただし現行robotorchanには多くのMixed variantがあるため、
**mixedであることは他の構造を捨てる理由ではありません**。

#### 13.26 「モデルが表現できる」と「BOで使いやすい」は別：実装上の補足

robotorchan では model / acquisition / objective / optimizer を分離しているため、これらを個別に検証します。

#### Registryは「自動的な正解表」ではない：実装上の補足

現行robotorchanのmodel registryには、mixed、multitask、inference方式、
high-dimensional strategy、robustness、multi-fidelity、structured-output、
preference、posterior sampling、fantasizationなどのcapability metadataがあります。

#### 13.40 自動モデル選択について：実装上の補足

robotorchan の将来的な拡張として、自動モデル候補生成も考えられます。

#### 13.40 自動モデル選択について：実装上の表現

```text
schema-based candidate suggestion
+
user-defined semantic metadata
```

#### 13.44 この章で覚えておくこと：実装上の補足


ここまでの理論章と [`docs/models.md`](../models.md)、実行 Notebook を併用することで、理論・API・実装例を相互に参照しながらモデルを選択できます。

#### 次に読むもの：実装上の補足

- 実際のモデル API と使い所: [`docs/models.md`](../models.md)
- 実行可能な例: [`examples/notebooks/`](../../examples/notebooks)
- BoTorch / robotorchan 内部設計: [`docs-internal/architecture/architecture.md`](../../docs-internal/architecture/architecture.md)

## 数値 API と利用上の補足

多くの Exact GP wrapper は

- 実行可能な例: [`examples/notebooks/`](../../examples/notebooks)

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

したがって本章では、通常のscalar regressionなら `SingleTaskGP` を比較基準に置きつつ、
**観測形式や構造が最初から異なる問題を無理にSingleTaskGPへ変換しない**方針を採用します。

```text
まず SingleTaskGP を考える
        ↓
標準 GP では表現できない構造がある？
        ↓
      Yes
        ↓
その構造を直接表現するモデルへ移る
```


この「なぜ標準 GP では足りないのか」を言語化できない場合、まず `SingleTaskGP` をベースラインにするのが安全です。

```text
観測データは何か？
│
├─ pairwise comparison
│    └─ PairwiseGP
│
└─ numeric response
     │
     ├─ structured output？
     │    ├─ tensor / grid → HigherOrderGP
     │    └─ explicit output coordinate T → LatentKroneckerGP
     │
     └─ scalar / ordinary output
          │
          ├─ conditional hierarchy？
          │    ├─ single task → HierarchicalConditionalKernelGP
          │    └─ multi-task → HierarchicalConditionalKernelMultiTaskGP
          │
          ├─ context decomposition？
          │    ├─ aggregated reward, independent context effect → SACGP
          │    ├─ aggregated reward, learned context similarity → LCEAGP
          │    └─ context-wise outputs observed → LCEMGP
          │
          ├─ multi-task / multi-output？
          │    ├─ independent outputs → ModelListGP
          │    ├─ same X for all tasks → KroneckerMultiTaskGP
          │    ├─ different X, same feature structure → MultiTaskGP
          │    └─ task-specific feature structure → HeterogeneousMTGP
          │
          ├─ fidelity dimension？
          │    └─ SingleTaskMultiFidelityGP
          │
          ├─ categorical variable？
          │    └─ MixedSingleTaskGP
          │
          └─ continuous ordinary regression
               │
               ├─ high dimensional？
               │    ├─ sparse effective dimension → SAAS
               │    ├─ lighter SAAS approximation → MAP-SAAS
               │    └─ additive structure → OrthogonalAdditiveGP
               │
               ├─ many observations？
               │    └─ SingleTaskVariationalGP
               │
               ├─ strong sparse outliers？
               │    └─ RobustRelevancePursuitSingleTaskGP
               │
               └─ otherwise → SingleTaskGP
```


```text
PairwiseGP
```


### HigherOrderGP

```text
train_Y
[n, d1, d2, ...]
```


### LatentKroneckerGP

```text
MixedSingleTaskGP
```


```text
HierarchicalConditionalKernelGP
```


```text
HierarchicalConditionalKernelMultiTaskGP
```


```text
SingleTaskMultiFidelityGP
```


複数の response があるからといって、必ず MultiTaskGP を使うわけではありません。

```text
ModelListGP
```


```text
train_X: [n, d]
train_Y: [n, m]
```


```text
KroneckerMultiTaskGP
```


```text
MultiTaskGP
```


```text
HeterogeneousMTGP
```



を同時最適化する場合、2つの目的を `ModelListGP` で独立にモデル化して qEHVI を使うこともできます。

```text
Multi-objective = MultiTaskGP
```


```text
SACGP
```


```text
LCEAGP
```


```text
LCEMGP
```


```text
SaasFullyBayesianSingleTaskGP
```


```text
SaasFullyBayesianMultiTaskGP
```


```text
AdditiveMapSaasSingleTaskGP
EnsembleMapSaasSingleTaskGP
```


```text
OrthogonalAdditiveGP
```


```text
SingleTaskVariationalGP
```


~~~text
既知・固定の観測noise
  → train_Yvar / likelihood

入力位置によってnoise varianceが変わる
  → heteroskedastic model

replicateからnoiseを推定できる
  → replicate-noise model

少数のcontamination / outlier
  → robust / contaminated model

入力そのものが揺らぐ
  → uncertain-input model

関数のsmoothnessやlengthscaleが場所で変わる
  → nonstationary model
~~~


```text
noise
→ likelihood / train_Yvar

outlier
→ robust / contaminated model

input uncertainty
→ uncertain-input model

nonstationarity
→ nonstationary model

regime
→ task / categorical / hierarchical model
```


### SingleTaskGP

```text
train_X: [n, d]
train_Y: [n, 1]
```



### KroneckerMultiTaskGP

```text
train_X: [n, d]
train_Y: [n, m]
```



### MultiTaskGP

```text
train_X: [n, d + 1]
                 ↑ task feature
train_Y: [n, 1]
```



### ModelListGP


### HeterogeneousMTGP

```text
train_Xs = [X_task0, X_task1, ...]
```


### HigherOrderGP

```text
train_X: [n, d]
train_Y: [n, d1, d2, ...]
```



### LatentKroneckerGP

```text
train_X
train_T
train_Y
```


### PairwiseGP

```text
SingleTaskGP
```


```text
MixedSingleTaskGP
```


```text
HierarchicalConditionalKernelGP
```


```text
SingleTaskMultiFidelityGP
```


```text
MultiTaskGP
```


```text
HeterogeneousMTGP
```


```text
PairwiseGP
```


```text
LatentKroneckerGP
```


```text
HigherOrderGP
```


```text
SaasFullyBayesianSingleTaskGP
```


```text
SingleTaskVariationalGP
```


特殊モデルを採用するときも、可能なら `SingleTaskGP` と比較することを推奨します。

```text
SingleTaskGP
vs
SAAS
```


```text
independent SingleTaskGP / ModelListGP
vs
MultiTaskGP
```


```text
ordinary GP
vs
SACGP / LCEAGP
```


```text
SingleTaskGP + LogEI
SingleTaskGP + UCB
SingleTaskGP + KG
```


```text
ModelListGP + qLogEHVI
```


```text
MixedSingleTaskGP
```


```text
SingleTaskVariationalGP
```


```text
SaasFullyBayesianSingleTaskGP
SaasFullyBayesianMultiTaskGP
```


```text
PairwiseGP
```


```text
MixedSingleTaskGP
```


## 13.30 よくある誤選択 2: multi-objective だから MultiTaskGP

```text
ModelListGP
```


MultiTaskGP は「複数目的だから」ではなく

Task A には存在しない feature を 0 埋めして通常 MultiTaskGP へ入れると、0 という値に物理的意味があるかのように扱われる可能性があります。

```text
HeterogeneousMTGP
```


```text
HierarchicalConditionalKernelGP
```


```text
LatentKroneckerGP
```


| 問題構造 | 基本候補 | 主な理由 |
|---|---|---|
| 通常の連続単一出力 | `SingleTaskGP` | 基準モデル |
| 連続 + categorical | `MixedSingleTaskGP` | カテゴリ距離を適切に扱う |
| fidelity | `SingleTaskMultiFidelityGP` | cost / fidelity correlation |
| related tasks | `MultiTaskGP` | task covariance |
| same X for every task | `KroneckerMultiTaskGP` | block / Kronecker structure |
| independent outputs | `ModelListGP` | 不要な相関を仮定しない |
| task-specific features | `HeterogeneousMTGP` | heterogeneous feature spaces |
| many observations | `SingleTaskVariationalGP` | inducing-point approximation |
| preference comparisons | `PairwiseGP` | latent utility model |
| high-d sparse effects | `SaasFullyBayesianSingleTaskGP` | sparse effective dimension |
| high-d multitask | `SaasFullyBayesianMultiTaskGP` | SAAS + task sharing |
| lighter SAAS | MAP-SAAS models | NUTS を避ける |
| additive high-d | `OrthogonalAdditiveGP` | additive structure |
| sparse outliers | `RobustRelevancePursuitSingleTaskGP` | robust modeling |
| tensor output | `HigherOrderGP` | output tensor covariance |
| explicit output axis | `LatentKroneckerGP` | X × T product space |
| conditional variables | `HierarchicalConditionalKernelGP` | active dimensions |
| conditional + multitask | hierarchical conditional multi-task GP | hierarchy + task covariance |
| contextual additive reward | `SACGP` | context decomposition |
| latent context similarity | `LCEAGP` | learned embeddings |
| context-wise outputs | `LCEMGP` | contextual multi-output |

```text
軽い
│
├─ SingleTaskGP
├─ MixedSingleTaskGP
├─ ModelListGP（小規模）
├─ MultiTaskGP（小規模）
│
├─ structured / hierarchical / contextual exact models
│
├─ Variational GP（大規模向けだが最適化設定が増える）
│
├─ MAP-SAAS
│
└─ Fully Bayesian SAAS / NUTS

重い
```


```text
SingleTaskGP
```


```text
SAAS
MultiTaskGP
VariationalGP
```


```text
categorical columns detected
→ MixedSingleTaskGP candidate

task feature exists
→ MultiTaskGP candidate

very high d
→ SAAS candidate

large n
→ VariationalGP candidate
```


```text
SingleTaskGP で十分なら SingleTaskGP を使う
```


```text
普通の連続回帰
→ SingleTaskGP

categorical
→ MixedSingleTaskGP

fidelity
→ SingleTaskMultiFidelityGP

related tasks
→ MultiTaskGP / KroneckerMultiTaskGP

independent outputs
→ ModelListGP

task-specific feature spaces
→ HeterogeneousMTGP

large n
→ SingleTaskVariationalGP

preference
→ PairwiseGP

high d + sparse effective dimensions
→ SAAS / MAP-SAAS

additive high-dimensional structure
→ OrthogonalAdditiveGP

sparse outliers
→ RobustRelevancePursuitSingleTaskGP

tensor output
→ HigherOrderGP

explicit output coordinates
→ LatentKroneckerGP

conditional active variables
→ HierarchicalConditionalKernelGP

conditional + multi-task
→ HierarchicalConditionalKernelMultiTaskGP

context decomposition + aggregated reward
→ SACGP / LCEAGP

context-wise output
→ LCEMGP
```

## 専用処理の補足

周辺尤度を最大化してカーネルなどのハイパーパラメータを推定します。
具体的な学習APIは各モデルの実装ガイドを参照してください。

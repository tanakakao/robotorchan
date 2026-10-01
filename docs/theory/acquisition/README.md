# Acquisition Function Theory

このディレクトリでは、獲得関数を「実装の使い方」ではなく、意思決定基準としての理論から整理します。

[Acquisition Function 概要](../04_acquisition_function.md) は獲得関数全体への入口です。
個別手法の数式、仮定、手法間の関係はこのディレクトリで扱います。
BoTorch / robotorchan の具体的な API、利用例、現在の対応範囲は
[Optimization guide](../../optimization/README.md) と
[Acquisition integration status](../../optimization/acquisition-integration.md) を参照してください。

## 責務

獲得関数関連のドキュメントは次の責務に分離します。

```text
docs/theory/04_acquisition_function.md
    獲得関数の概要と詳細理論への入口

docs/theory/acquisition/
    数式、直感、仮定、導出、手法間の関係

docs/optimization/
    BoTorch / robotorchan での利用方法、API、実装上の制約

docs/optimization/acquisition-integration.md
    現在の実装・統合状況

robotorchan/acquisition/
    robotorchan が所有する実装
```

Theory に掲載されていることは robotorchan での実装を意味しません。逆に、BoTorch native の手法も理論上重要であれば Theory で扱います。

## 最初に理解する3つの軸

獲得関数をモデル名のように暗記する前に、次の3軸を分けます。

| 軸 | 問い | 例 |
| --- | --- | --- |
| decision purpose | 何を達成したいか | BO、Active Learning、Level-set |
| utility principle | 何を価値とみなすか | improvement、confidence、information、lookahead |
| posterior requirement | modelから何が必要か | marginal moments、joint Gaussian、posterior samples |

例えば「Monte Carlo acquisition」は目的ではなく**計算・期待値評価の方式**です。
「q-acquisition」も目的ではなく**複数candidateをjointに評価する構造**です。
この2つをEI、KG、EHVI等と同じ分類軸として扱いません。

## 章構成

| 章 | 主題 | 主な内容 |
| --- | --- | --- |
| [01](01_foundations.md) | Foundations | posterior、utility、exploration / exploitation |
| [02](02_improvement.md) | Improvement | EI、LogEI、PI、LogPI と improvement-based utility |
| [03](03_confidence_bound.md) | Confidence Bound | UCB、qUCB、confidence parameter |
| [04](04_batch_noisy.md) | Batch / Noisy | q-acquisition、joint utility、correlation、NEI、pending |
| [05](05_information_theoretic.md) | Information Theory | entropy、mutual information、MES、GIBBON |
| [06](06_lookahead.md) | Lookahead | KG、qKG、fantasy model、multi-step lookahead |
| [07](07_multiobjective.md) | Multi-objective | Pareto、hypervolume、EHVI、NEHVI、NParEGO、HVKG |
| [08](08_constraints.md) | Constraints | feasibility、constrained utility、objective / transform |
| [09](09_active_learning.md) | Active Learning | variance / std、integrated variance、qNIPV、EPIG |
| [10](10_level_set.md) | Level-set | level-set、Straddle、Randomized Straddle、BoundaryVariance |
| [11](11_multifidelity_cost_aware.md) | Multi-Fidelity / Cost-aware | fidelity、VoI、MF-KG、cost |
| [12](12_selection_guide.md) | Selection Guide | 問題設定と獲得関数ファミリーの対応、各詳細章への導線 |

## 初心者向けの推奨ルート

最初から12章を番号順に暗記する必要はありません。標準的な単目的BOから入る場合は、

~~~text
01 Foundations
 -> 02 Improvement
 -> 03 Confidence Bound
 -> 04 Batch / Noisy
~~~

までで、posteriorからcandidateを選ぶ基本構造を学べます。

その後は目的に応じて分岐します。

~~~text
最適値について情報を集める       -> 05 Information Theory
観測後の意思決定価値を見る       -> 06 Lookahead
複数目的を同時に扱う             -> 07 Multi-objective
出力側の制約を考慮する           -> 08 Constraints
関数全体・予測を学習する         -> 09 Active Learning
境界・閾値領域を学習する         -> 10 Level-set
fidelityと評価costを考慮する      -> 11 Multi-Fidelity / Cost-aware
問題から逆引きする               -> 12 Selection Guide
~~~

この順序は性能順位ではなく、概念依存関係を理解しやすくするためのnavigationです。

## 共通記述方針

各章は内容に応じて、次の順序を基本とします。

1. **目的と直感** — 何を知る、または最適化するための基準か。
2. **問題設定** — 入力、posterior、目的量、必要な仮定を定義する。
3. **数学的定義** — 記号を定義し、主要な式を示す。
4. **理論的意味** — 式の各項、探索挙動、情報価値を説明する。
5. **拡張と関係** — batch、noise、multi-output などへの拡張と関連手法を整理する。
6. **適用範囲と注意点** — 成立条件、計算量、失敗しやすい条件を示す。
7. **BoTorch / robotorchan との対応** — 実装詳細を再説明せず、対応先だけを明示する。
8. **参考文献** — 原論文、主要な理論文献、必要に応じて BoTorch の一次資料を示す。

全章へ機械的に同じ節を置く必要はありません。例えば Pareto 理論や entropy の説明では、その理論に自然な構成を優先します。

## 記述ルール

### Theory と API を混在させない

Theory の本文は、特定のラッパー API の説明を中心にしません。コード例や引数仕様は `docs/optimization/` の責務です。

実装への対応を示す場合も、

```text
理論: Expected Improvement
BoTorch: LogExpectedImprovement / qLogExpectedImprovement
robotorchan: native BoTorch path を利用
```

のように、理論と実装の境界を明示します。

### 実装済み手法だけに限定しない

Theory は理論体系として構成します。robotorchan にローカル実装がない MES、GIBBON、qMultiStepLookahead なども、理論上の位置付けが重要なら扱います。

現在の対応状況はTheory本文から推測せず、
[Acquisition integration status](../../optimization/acquisition-integration.md) とexecutable runtime testsを
確認します。registryは重要なstatic compatibility surfaceですが、runtime validationの代替ではありません。

### 原典を優先する

数式、アルゴリズムの定義、名称の由来は、可能な限り原論文または一次資料を基準にします。robotorchan の実装から理論定義を逆算しません。

robotorchan 固有の手法は固有であることを明示します。例えば `BoundaryVariance` は標準的な文献手法として扱いません。

### maximize / minimize を曖昧にしない

式では最大化・最小化の convention を明示します。符号反転や objective transform が必要な場合は、その前提を説明します。

### analytic / MC と posterior requirement を区別する

analytic / Monte Carloはacquisition valueをどう評価するかという分類です。
一方、posterior requirementはmodelからどの情報が必要かという互換性の分類です。

robotorchanのcapability metadataでは、主に次を区別します。

- `MARGINAL_MOMENTS`: candidateごとのmean / variance等
- `JOINT_GAUSSIAN`: candidate間を含むjoint Gaussian structure
- `POSTERIOR_SAMPLES`: acquisitionが必要とするjoint posterior samples

したがって「MCだから任意のsampleable modelで使える」「Gaussianだからanalytic acquisitionが使える」
とは自動的に判断しません。output、q、fantasization等の追加requirementsも確認します。

### q と batch の意味を区別する

q-acquisition を「1点用スコアの上位 q 点」として説明しません。joint posterior と候補間相関を扱う基準であることを明確にします。

### noise の種類を区別する

latent function uncertainty、observation noise、model uncertainty を必要に応じて区別します。
Noisy acquisition の説明では、観測値の最大値と潜在関数の最良値を混同しません。

### 情報獲得の対象を明示する

Information-theoretic acquisition では「何についての情報量か」を必ず明示します。

- MES / GIBBON: optimum value に関する情報。
- EPIG: target distribution 上の prediction に関する情報。
- KG: 観測後の意思決定価値。

これらを単に「情報量を最大化する手法」と一括りにしません。

### constraintを2種類に分ける

acquisition theoryで扱うoutcome / black-box constraintと、candidate optimizerが扱う
input-space constraintは別の責務です。

~~~text
unknown feasibility learned from outcomes
    -> acquisition / objective側

known bounds, equality, inequality on candidate X
    -> candidate optimization側
~~~

「constrained acquisition対応」という表現だけから、既知のcandidate constraintまで対応すると
推論しません。

### BO / Active Learning / Level-set を区別する

```text
Bayesian Optimization
    最適な入力または目的値を見つける

Active Learning
    予測分布や関数について効率よく学習する

Level-set Estimation
    f(x) = t となる境界や領域を学習する
```

目的が異なるため、同じ posterior uncertainty を利用していても獲得基準の意味は異なります。

### 開発履歴を理論本文へ持ち込まない

`Phase N`、PR番号、過去APIなどの開発履歴はTheory本文に記載しません。Theoryは現在の理論体系を説明します。

## 04 Acquisition Function との境界

[04 Acquisition Function](../04_acquisition_function.md) は、今後次の内容に集中させます。

- acquisition function の役割。
- posterior から candidate selection までの流れ。
- exploration と exploitation。
- analytic と Monte Carlo の違い。
- sequential / batch / noisy の概念。
- BO、Active Learning、Level-set の目的の違い。
- acquisition function と objective / optimizer / surrogate model の責務分離。
- 本ディレクトリへのナビゲーション。

EI、UCB、KG、EHVI などの詳細な式・導出・比較は、それぞれの詳細章へ移します。

## Optimization guide との境界

`docs/optimization/` は、次を担当します。

- import path と API。
- 実行可能なコード例。
- q=1 / q>1、multi-output、ensemble など現在の実装制約。
- BoTorch native path と robotorchan-owned implementation の区別。
- candidate optimization の具体的方法。

同じ数式や理論説明を両方へ複製せず、Optimization guide から対応する Theory 章へリンクします。

## 理論名と現在の実装supportを分離する

Theoryには、BoTorch native、robotorchan-owned、理論上重要だが現在の統合対象外の手法が
同時に登場します。

そのため各章では、

~~~text
理論として定義できる
    != BoTorchに実装がある
    != robotorchan registryに登録されている
    != 任意のrobotorchan modelと互換
~~~

と考えます。

現在のrobotorchan registryは、purpose、posterior requirement、最大q、multi-output、
structured-output、ensemble、constraint、multi-objective、fantasization、multi-fidelity、
one-shot等を機械可読なcapabilityとして持ちます。

Theory本文はこのmetadataのコピーにはせず、**なぜそのrequirementが必要になるのか**を説明します。

## 読み方

初めて読む場合は [Foundations](01_foundations.md) から順に進むと、標準 BO から高度な意思決定問題まで段階的に理解できます。

実際の問題から獲得関数を選びたい場合は、先に [Selection Guide](12_selection_guide.md) を読み、必要な詳細章へ移動してください。

実装方法を確認したい場合は [Optimization guide](../../optimization/README.md)、
現在の対応範囲だけを確認したい場合は
[Acquisition integration status](../../optimization/acquisition-integration.md) を参照してください。


## 各章で確認する共通チェック

各acquisition familyの監査では、少なくとも次を確認します。

1. 最適化・学習したい量が明示されているか
2. maximize / minimize conventionが明示されているか
3. posteriorの何を使うかが説明されているか
4. q=1とq>1を機械的に同一視していないか
5. noise / pending / fantasyの役割を混同していないか
6. single-output / multi-output / multi-objectiveを区別しているか
7. outcome constraintとcandidate constraintを区別しているか
8. analytic / MC / one-shot等の計算構造を説明しているか
9. BoTorch nativeとrobotorchan-owned implementationのprovenanceが明確か
10. 原論文・canonical referenceへ辿れるか

## Documentation QA contract

この acquisition theory hierarchy は次の整合性を維持します。

- chapter number と filename は 01〜12 で一致させる。
- theoryからimplementation supportを推測しない。現在の対応範囲は
  [Acquisition integration status](../../optimization/acquisition-integration.md)、registry、runtime testsを
  相互確認する。metadataとruntimeが不一致なら実行時contractを優先してgapを修正する。
- practical API の説明は [Optimization guide](../../optimization/README.md) に置き、Theory へ
  implementation-specific usage を重複させない。
- BoTorch native acquisition は、robotorchan 固有 contract がない限り local alias として
  再公開しない。
- literature method と robotorchan-specific heuristic の provenance を混同しない。
- q=1、multi-output、structured output、ensemble posterior などの制約を Theory の一般形から
  自動的に拡張しない。



## Cross-chapter invariants

章をまたいで次の意味を固定します。

~~~text
q-batch       = 複数のreal candidateをjointに価値付けする構造
X_pending     = 実際に評価中で結果が未確定のcandidate context
fantasy       = hypothetical outcomeでconditionしたmodel / branch
one-shot rows = lookahead acquisition内部のauxiliary decision variables
~~~

これらは互いに置換しません。同様に、multi-output modelとmulti-objective decision、output constraintと
candidate constraint、cost-aware utilityとmulti-fidelity information transferを別axisとして扱います。

posterior requirementも `MARGINAL_MOMENTS`、`JOINT_GAUSSIAN`、`POSTERIOR_SAMPLES`を区別し、
「sampleableなら任意のMC acquisitionに互換」とは推論しません。

## Cross-layer verification order

~~~text
problem semantics
 -> theory family
 -> model / posterior contract
 -> acquisition capability
 -> optimizer / search-space contract
 -> executable runtime validation
~~~

Theory coverage、BoTorch native API、robotorchan registry entry、robotorchan-owned implementation、
runtime-tested workflowは同義ではありません。

## Maintenance state

Acquisition theory の初期体系化は完了しています。以後は phase-based な構築履歴ではなく、
通常の保守対象として扱います。

変更時は、必要な層だけを更新します。

```text
theory / equations / provenance
    -> docs/theory/acquisition/

usage / API / examples
    -> docs/optimization/

current support / explicit limitations
    -> docs/optimization/acquisition-integration.md

robotorchan-owned behavior
    -> src/robotorchan/acquisition/ + tests
```

新しい獲得関数を検討するときは、まず BoTorch native path の有無を確認し、Theory 掲載、
integration coverage、robotorchan-owned implementation を別々の判断として扱います。

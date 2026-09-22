# Acquisition Function Theory

このディレクトリでは、獲得関数を「実装の使い方」ではなく、意思決定基準としての理論から整理します。

[Acquisition Function 概要](../04_acquisition_function.md) は獲得関数全体への入口です。個別手法の数式、仮定、手法間の関係はこのディレクトリで扱います。BoTorch / robotorchan の具体的な API、利用例、現在の対応範囲は [Optimization guide](../../optimization/README.md) と [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 責務

獲得関数関連のドキュメントは次の責務に分離します。

```text
docs/theory/04_acquisition_function.md
    獲得関数の概要と詳細理論への入口

docs/theory/acquisition/
    数式、直感、仮定、導出、手法間の関係

docs/optimization/
    BoTorch / robotorchan での利用方法、API、実装上の制約

docs/optimization/acquisition-status.md
    現在の実装・統合状況

robotorchan/acquisition/
    robotorchan が所有する実装
```

Theory に掲載されていることは robotorchan での実装を意味しません。逆に、BoTorch native の手法も理論上重要であれば Theory で扱います。

## 章構成

| 章 | 主題 | 主な内容 |
| --- | --- | --- |
| [01](01_foundations.md) | Foundations | posterior、utility、exploration / exploitation、analytic / MC、acquisition optimization |
| [02](02_improvement.md) | Improvement | EI、LogEI、PI、LogPI と improvement-based utility |
| [03](03_confidence_bound.md) | Confidence Bound | UCB、qUCB、confidence parameter |
| [04](04_batch_noisy.md) | Batch / Noisy | q-acquisition、joint utility、posterior correlation、NEI、pending points |
| [05](05_information_theoretic.md) | Information Theory | entropy、mutual information、MES、GIBBON |
| [06](06_lookahead.md) | Lookahead | KG、qKG、fantasy model、multi-step lookahead |
| [07](07_multiobjective.md) | Multi-objective | Pareto dominance、hypervolume、EHVI、NEHVI、NParEGO、HVKG |
| [08](08_constraints.md) | Constraints | feasibility、constraint-aware acquisition、objective / posterior transform |
| [09](09_active_learning.md) | Active Learning | posterior variance / std、integrated variance、qNIPV、EPIG |
| [10](10_level_set.md) | Level-set | level-set estimation、Straddle、Randomized Straddle、BoundaryVariance |
| [11](11_multifidelity_cost_aware.md) | Multi-Fidelity / Cost-aware | fidelity、value of information、MF-KG、cost-aware utility |
| [12](12_selection_guide.md) | Selection Guide | 問題設定と獲得関数ファミリーの対応、各詳細章への導線 |

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

現在の対応状況は Theory ではなく [Acquisition integration status](../../optimization/acquisition-status.md) を正とします。

### 原典を優先する

数式、アルゴリズムの定義、名称の由来は、可能な限り原論文または一次資料を基準にします。robotorchan の実装から理論定義を逆算しません。

robotorchan 固有の手法は固有であることを明示します。例えば `BoundaryVariance` は標準的な文献手法として扱いません。

### maximize / minimize を曖昧にしない

式では最大化・最小化の convention を明示します。符号反転や objective transform が必要な場合は、その前提を説明します。

### q と batch の意味を区別する

q-acquisition を「1点用スコアの上位 q 点」として説明しません。joint posterior と候補間相関を扱う基準であることを明確にします。

### noise の種類を区別する

latent function uncertainty、observation noise、model uncertainty を必要に応じて区別します。Noisy acquisition の説明では、観測値の最大値と潜在関数の最良値を混同しません。

### 情報獲得の対象を明示する

Information-theoretic acquisition では「何についての情報量か」を必ず明示します。

- MES / GIBBON: optimum value に関する情報。
- EPIG: target distribution 上の prediction に関する情報。
- KG: 観測後の意思決定価値。

これらを単に「情報量を最大化する手法」と一括りにしません。

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

## 読み方

初めて読む場合は [Foundations](01_foundations.md) から順に進むと、標準 BO から高度な意思決定問題まで段階的に理解できます。

実際の問題から獲得関数を選びたい場合は、先に [Selection Guide](12_selection_guide.md) を読み、必要な詳細章へ移動してください。

実装方法を確認したい場合は [Optimization guide](../../optimization/README.md)、現在の対応範囲だけを確認したい場合は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

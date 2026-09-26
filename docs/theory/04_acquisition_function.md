# 4. Acquisition Function

## 4.1 Acquisition Function とは

ベイズ最適化では、Gaussian Process（GP）などの surrogate model から得られる
posterior distribution を使って、**次にどこを評価するか**を決めます。

現在までの観測データを

\[
\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n
\]

とし、候補点 \(x\) に対する獲得関数を

\[
\alpha(x; \mathcal{D}_n)
\]

と書くと、逐次評価では典型的に

\[
x_{n+1}
=
\operatorname*{arg\,max}_{x \in \mathcal{X}}
\alpha(x; \mathcal{D}_n)
\]

として次の候補を選びます。

獲得関数は、surrogate model の予測そのものではありません。posterior に含まれる
予測値、不確実性、相関構造などを、**次の観測の価値**へ変換する意思決定層です。

```text
observations
    ↓
surrogate model
    ↓
posterior distribution
    ↓
objective / utility
    ↓
acquisition function
    ↓
acquisition optimization
    ↓
candidate
```

個別手法の詳細な数式と理論は
[Acquisition Function Theory](acquisition/README.md) に分離しています。本章では、
それらを読むために必要な全体像を整理します。

## 4.2 なぜ posterior mean だけでは不十分か

posterior mean \(\mu_n(x)\) だけを最大化すると、現在のモデルがすでに良いと
考えている場所へ評価が集中しやすくなります。

一方、未観測領域では posterior uncertainty が大きく、現在の予測を更新する価値が
残っています。獲得関数はこのような情報を使って、典型的には次の二つを扱います。

- **Exploitation（活用）**: 現在良いと予測される場所を評価する。
- **Exploration（探索）**: 不確実で、追加情報の価値がある場所を評価する。

ただし、すべての獲得関数がこの二項を明示的に足し合わせるわけではありません。
improvement、confidence bound、information gain、value of information など、
「観測価値」の定義そのものが手法ごとに異なります。

## 4.3 Posterior から観測価値を定義する

単一出力 GP では、候補点 \(x\) に対して典型的に

\[
f(x) \mid \mathcal{D}_n
\sim
\mathcal{N}(\mu_n(x), \sigma_n^2(x))
\]

という posterior が得られます。

代表的な獲得関数ファミリーは、この posterior に対して異なる問いを設定します。

| ファミリー | 主な問い |
| --- | --- |
| Improvement | 現在よりどれだけ改善できそうか |
| Confidence bound | 平均と不確実性をどう組み合わせるか |
| Information-theoretic | 最適値など特定の未知量についてどれだけ情報を得られるか |
| Lookahead | 観測後の将来の意思決定がどれだけ改善するか |
| Multi-objective | Pareto front や hypervolume をどれだけ改善できるか |
| Active Learning | 関数や予測分布をどれだけ効率よく学習できるか |
| Level-set | 指定した応答レベルの境界をどれだけ効率よく学習できるか |
| Multi-Fidelity | 情報価値と評価 fidelity / cost をどう両立するか |

この違いを理解することが、個別の獲得関数名を暗記するより重要です。

## 4.4 代表的な単目的 BO の考え方

標準的な単目的 BO では、Improvement と Confidence Bound が基本的な入口になります。

Expected Improvement（EI）は、最大化問題なら概念的に

\[
I(x) = \max(f(x)-f_{\mathrm{best}}, 0)
\]

という improvement の期待値を評価します。Probability of Improvement（PI）は改善量
ではなく改善確率を評価します。

Upper Confidence Bound（UCB）は、posterior mean と uncertainty を直接組み合わせる
考え方です。

\[
\operatorname{UCB}(x)
=
\mu_n(x)+\sqrt{\beta}\,\sigma_n(x)
\]

これらの導出、LogEI / LogPI、パラメータの意味は詳細章で扱います。

- [Improvement-based acquisition](acquisition/02_improvement.md)
- [Confidence-bound acquisition](acquisition/03_confidence_bound.md)

## 4.5 Analytic と Monte Carlo

獲得関数には、posterior に対する期待値を閉形式で計算できる analytic acquisition と、
posterior samples を使って期待効用を近似する Monte Carlo（MC）acquisition があります。

MC では概念的に

\[
\alpha(X)
\approx
\frac{1}{S}
\sum_{s=1}^{S}
U(f^{(s)}(X))
\]

のように評価します。

MC は batch、multi-output、nonlinear objective、constraints などへ拡張しやすく、
BoTorch の獲得関数設計でも重要な役割を持ちます。

詳細は [Foundations](acquisition/01_foundations.md) で扱います。

## 4.6 Sequential、Batch、Noisy

1点ずつ評価する逐次 BO だけでなく、複数点を同時に評価する batch BO があります。
候補集合を

\[
X = \{x_1,\ldots,x_q\}
\]

とすると、q-acquisition は通常、各点の独立スコア上位を選ぶものではありません。
**候補集合の joint posterior と候補間相関を含めた価値**を評価します。

また、観測ノイズがある場合には、観測された最大値と潜在関数の真の最良値を区別する
必要があります。実行中で結果が未取得の pending points も、非同期 BO では候補選択へ
影響します。

これらは [Batch and noisy acquisition](acquisition/04_batch_noisy.md) でまとめて扱います。

## 4.7 情報量と Lookahead は同じではない

より高度な獲得関数では「情報」を扱いますが、何についての情報かを区別する必要が
あります。

Max-value Entropy Search（MES）などは、最適値のような未知量について得られる情報を
基準にします。

Knowledge Gradient（KG）は、候補を観測した後に最終的な意思決定の価値がどれだけ
改善するかという value of information を扱います。さらに multi-step lookahead では、
複数段階先の意思決定を fantasy model によって評価します。

- [Information-theoretic acquisition](acquisition/05_information_theoretic.md)
- [Lookahead acquisition](acquisition/06_lookahead.md)

## 4.8 Multi-Objective Bayesian Optimization

複数目的

\[
f(x)=(f_1(x),\ldots,f_m(x))
\]

を同時に扱う場合、単一の最良値だけではなく Pareto dominance や Pareto front を
考えます。

Hypervolume-based acquisition では reference point に対する hypervolume improvement
を観測価値として使います。EHVI / NEHVI はこの考え方に基づきます。一方、
NParEGO のように scalarization を利用する方法や、lookahead を組み合わせる方法も
あります。

詳細は [Multi-objective acquisition](acquisition/07_multiobjective.md) で扱います。

## 4.9 Constraints と Objective

実問題では、目的関数だけでなく

\[
g_j(x) \le 0
\]

のような制約を満たす必要があります。制約付き BO では、目的改善と feasibility を
組み合わせて候補の価値を評価します。

また、**model output、objective、acquisition function は別の責務**です。
multi-output model から得た posterior sample を scalar objective へ変換してから
utility を計算する場合もあります。

```text
model posterior
    ↓
objective / posterior transform
    ↓
utility
    ↓
acquisition value
```

詳細は [Constrained acquisition](acquisition/08_constraints.md) で扱います。

ここでいう制約は、未知の応答を surrogate model で推定する **output / black-box constraint** です。一方、組成和、変数間の大小関係、既知の線形設計則など、candidate の入力座標そのものに課す制約は acquisition function ではなく acquisition optimization の責務です。

```text
output constraint:     g(x) <= 0  → model / acquisition composition
candidate constraint:  A x <= b   → acquisition optimizer
```

robotorchan では後者を対応strategyの `CandidateConstraints` で表現します。この二種類を同じ「constrained BO」として扱わないことが、APIと実装を読む際に重要です。

## 4.10 Bayesian Optimization だけが目的ではない

posterior を使った逐次実験設計は、最適化だけに限定されません。

### Bayesian Optimization

目的は、良い入力、最適な入力、または最適目的値を効率よく見つけることです。

### Active Learning

目的は、関数や予測分布について効率よく学習することです。posterior variance、
integrated variance reduction、predictive information gain などが基準になります。

### Level-set Estimation

目的は、指定した target \(t\) に対して

\[
f(x)=t
\]

となる境界や、その上下の領域を効率よく学習することです。Straddle のような基準は
この目的に対応します。

同じ posterior uncertainty を利用していても、最適化、予測学習、境界推定では
「良い候補」の意味が異なります。

- [Active Learning](acquisition/09_active_learning.md)
- [Level-set Estimation](acquisition/10_level_set.md)

## 4.11 Multi-Fidelity と Cost-aware Decision

評価 fidelity によって精度やコストが異なる問題では、常に最高 fidelity を評価する
ことが効率的とは限りません。

Multi-Fidelity BO では、

```text
その観測から得られる情報価値
                ×
fidelity / evaluation cost
```

という関係を考えます。Knowledge Gradient と cost-aware utility を組み合わせる
アプローチはその代表例です。

詳細は
[Multi-Fidelity and cost-aware acquisition](acquisition/11_multifidelity_cost_aware.md)
で扱います。

## 4.12 Acquisition Function とその最適化は別問題

獲得関数を定義することと、それを最大化することは別の問題です。

\[
\text{acquisition definition}
\neq
\text{acquisition optimization}
\]

連続変数では勾配ベース最適化を利用できる場合がありますが、categorical、integer、
hierarchical、discrete fidelity などを含むと candidate optimization 自体に追加の設計が
必要です。

これは surrogate model の Mixed Variables 対応とも別の論点です。探索空間については
次章の [Mixed Variables](05_mixed_variables.md) も参照してください。

## 4.13 Surrogate Model との相互作用

獲得関数は posterior を使って意思決定するため、posterior quality に依存します。

例えば uncertainty を過小評価する surrogate では探索が弱くなり、過大評価すれば
不要な探索が増える可能性があります。kernel、noise model、task structure、input
transform などの設計と獲得関数は、ソフトウェア上は責務を分離できても、統計的には
独立ではありません。

## 4.14 Theory、利用方法、実装状況を分けて読む

robotorchan では獲得関数関連の情報を次のように分離します。

| 知りたいこと | 参照先 |
| --- | --- |
| 獲得関数全体の概要 | 本章 |
| 数式、理論、仮定、手法間の関係 | [Acquisition Function Theory](acquisition/README.md) |
| API、コード例、実際の利用方法 | [Optimization guide](../optimization/README.md) |
| 現在の対応範囲と制約 | [Acquisition integration status](../optimization/acquisition-status.md) |

robotorchan は BoTorch-first を基本とします。Theory に掲載されている手法が
robotorchan 固有実装であるとは限らず、BoTorch native の手法を直接利用する場合も
あります。

## 4.15 詳細理論への入口

詳細理論は次の順序で整理します。

1. [Foundations](acquisition/01_foundations.md)
2. [Improvement](acquisition/02_improvement.md)
3. [Confidence Bound](acquisition/03_confidence_bound.md)
4. [Batch / Noisy](acquisition/04_batch_noisy.md)
5. [Information Theory](acquisition/05_information_theoretic.md)
6. [Lookahead](acquisition/06_lookahead.md)
7. [Multi-objective](acquisition/07_multiobjective.md)
8. [Constraints](acquisition/08_constraints.md)
9. [Active Learning](acquisition/09_active_learning.md)
10. [Level-set](acquisition/10_level_set.md)
11. [Multi-Fidelity / Cost-aware](acquisition/11_multifidelity_cost_aware.md)
12. [Selection Guide](acquisition/12_selection_guide.md)

詳細章では、現在の本章に含まれていた EI、PI、UCB、KG、EHVI、NEHVI などを
削除するのではなく、より体系的に移行・拡張します。

## 4.16 まとめ

Acquisition Function は、surrogate model の posterior を

**次に何を観測する価値があるか**

という意思決定へ変換する層です。

```text
Model
  ↓
Posterior
  ↓
Objective / Utility
  ↓
Acquisition Function
  ↓
Acquisition Optimization
  ↓
Candidate
```

重要なのは、個別の手法名だけでなく、

- 何を知る、または最適化したいのか。
- uncertainty をどのような価値へ変換するのか。
- sequential / batch / noisy のどの設定か。
- constraints、multi-objective、fidelity があるか。
- surrogate posterior がその意思決定に十分な品質か。

を区別することです。

次章では探索空間側の問題として
[Mixed Variables](05_mixed_variables.md) を扱います。獲得関数そのものの詳細は
[Acquisition Function Theory](acquisition/README.md) へ進んでください。

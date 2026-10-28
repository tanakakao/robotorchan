# 4. Acquisition Function

本章では、事後分布から期待効用を定義し、次の実験候補を選ぶまでの関係を学びます。
[GP章](02_gaussian_process.md)の予測平均・共分散と、期待値の基礎を前提とします。
最大化を基準とし、$x\in\mathcal X$ を一つの候補、$X$ を複数候補の組、
$n$ を取得済みの観測数とします。

## 4.1 Acquisition Function とは

ベイズ最適化では、Gaussian Process（GP）などの surrogate model から得られる
posterior distribution を使って、**次にどこを評価するか**を決めます。

現在までの観測データを

$$
\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n
$$

とし、候補点 $`x`$ に対する獲得関数を

$$
\alpha(x; \mathcal{D}_n)
$$

と書くと、逐次評価では典型的に

$$
x_{n+1}
=
\operatorname*{arg\,max}_{x \in \mathcal{X}}
\alpha(x; \mathcal{D}_n)
$$

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

### 予測と意思決定を分けて考える

例えば2つの候補A、Bについて、posteriorが次のような状態だとします。

~~~text
A: 予測平均は高いが、すでによく調べられていて不確実性が小さい
B: 予測平均は少し低いが、まだよく分からず不確実性が大きい
~~~

「予測値が最大の点」を選ぶだけならAを選びます。しかしBOでは、Bを観測することで
未知の良い領域を発見できる可能性も考慮します。**どちらをどの程度価値ある候補と
みなすかを数値化するのが acquisition function**です。

したがって acquisition value の単位や大きさを、そのまま目的関数の予測値として
解釈してはいけません。その意味は acquisition function ごとに異なります。

## 4.2 なぜ posterior mean だけでは不十分か

posterior mean $`\mu_n(x)`$ だけを最大化すると、現在のモデルがすでに良いと
考えている場所へ評価が集中しやすくなります。

一方、未観測領域では posterior uncertainty が大きく、現在の予測を更新する価値が
残っています。獲得関数はこのような情報を使って、典型的には次の二つを扱います。

- **Exploitation（活用）**: 現在良いと予測される場所を評価する。
- **Exploration（探索）**: 不確実で、追加情報の価値がある場所を評価する。

ただし、すべての獲得関数がこの二項を明示的に足し合わせるわけではありません。
improvement、confidence bound、information gain、value of information など、
「観測価値」の定義そのものが手法ごとに異なります。

## 4.3 Posterior から観測価値を定義する

単一出力 GP では、候補点 $`x`$ に対して典型的に

$$
f(x) \mid \mathcal{D}_n
\sim
\mathcal{N}(\mu_n(x), \sigma_n^2(x))
$$

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

$$
I(x) = \max(f(x)-f_{\mathrm{best}}, 0)
$$

という improvement の期待値を評価します。つまり

$$
\operatorname{EI}(x)
=
\mathbb{E}\left[
\max(f(x)-f_{\mathrm{best}},0)
\mid \mathcal{D}_n
\right]
$$

です。ここで期待値を取るのは、$`f(x)`$ が posterior のもとではまだ未知の
確率変数だからです。EIは「改善する確率」だけでなく、
**改善した場合にどれくらい大きく改善するか**も同時に考慮します。

### 正規事後分布での期待改善量の導出

比較基準 $b=f_{\mathrm{best}}$ を既知の定数とし、
$f(x)=\mu_n(x)+\sigma_n(x)Z$、$Z\sim\mathcal N(0,1)$ と書きます。
$\sigma_n(x)>0$ のとき $z=(\mu_n(x)-b)/\sigma_n(x)$ と置くと、
改善が正となるのは $Z>-z$ の範囲です。標準正規分布の密度を $\phi$、累積分布を $\Phi$ として、

$$
\begin{aligned}
\operatorname{EI}(x)
&=\int_{-z}^{\infty}(\mu_n(x)-b+\sigma_n(x)t)\phi(t)\,dt\\
&=(\mu_n(x)-b)\Phi(z)+\sigma_n(x)\phi(z).
\end{aligned}
$$

最後の等号は $\phi'(t)=-t\phi(t)$ と正規分布の対称性から得られます。
$\sigma_n(x)=0$ では確定値の改善 $\max(\mu_n(x)-b,0)$ を使います。
例えば $\mu_n(x)=b$、$\sigma_n(x)=0.2$ ならEIは
$0.2/\sqrt{2\pi}\simeq0.0798$ で、予測平均が基準と同じでも正の価値があります。

Probability of Improvement（PI）は改善量ではなく改善確率を評価します。
上と同じ仮定で $\operatorname{PI}(x)=\Phi(z)$ です。
平均が基準と等しく分散が正ならPIは標準偏差によらず0.5ですが、EIは標準偏差に比例します。
これが改善する確率と改善量を評価することの違いです。
PIは厳密な改善 $f(x)>b$ の確率なので、分散ゼロで平均も基準と一致すれば0になります。

Upper Confidence Bound（UCB）は、posterior mean と uncertainty を直接組み合わせる
考え方です。

$$
\operatorname{UCB}(x)
=
\mu_n(x)+\sqrt{\beta}\,\sigma_n(x)
$$

ここで $\beta\ge0$ は不確実性への重みです。
正規分布の点ごとの上側分位点は $\mu_n(x)+\Phi^{-1}(1-\delta)\sigma_n(x)$ なので、
$0<\delta<1/2$ に対して $\sqrt\beta=\Phi^{-1}(1-\delta)$ と置くと同じ形になります。
ただし、この点ごとの解釈だけで全候補・全時刻を同時に覆う信頼保証は得られません。
LogEI / LogPIや信頼係数の設定の詳細は、次の各章で扱います。

- [Improvement-based acquisition](acquisition/02_improvement.md)
- [Confidence-bound acquisition](acquisition/03_confidence_bound.md)

### 最大化・最小化と符号

獲得関数の式は最大化問題を基準に説明されることが多いですが、実問題には最小化もあります。
重要なのは「獲得関数を最大化する」という外側の最適化と、
「目的関数を最大化したいか最小化したいか」を区別することです。

最小化なら効用を $\max(f_{\mathrm{best}}-f(x),0)$ とするか、
$g=-f$ として最大化へ向きをそろえます。
したがって、数式を利用するときは $`f_{\mathrm{best}}`$ の定義や improvement の符号が
最大化・最小化のどちらを前提としているか確認する必要があります。

## 4.5 Analytic と Monte Carlo

獲得関数には、posterior に対する期待値を閉形式で計算できる analytic acquisition と、
posterior samples を使って期待効用を近似する Monte Carlo（MC）acquisition があります。

MC では概念的に

$$
\alpha(X)
\approx
\frac{1}{S}
\sum_{s=1}^{S}
U(f^{(s)}(X))
$$

のように評価します。ここで $S$ は標本数、$f^{(s)}(X)$ は
候補集合 $X$ の**同時事後分布**からの第 $s$ 標本、$U$ は集合全体の効用です。
例えば $q$ 点の期待改善量なら $U(v)=\max(\max_{i\le q}v_i-b,0)$ と置けます。
各候補を独立に標本化すると、相関を含む本来の同時分布とは異なる期待値になります。

独立同分布の標本と有限な効用分散を仮定すれば、標本平均は不偏であり、
標準誤差は $S^{-1/2}$ に比例します。有限標本で最大化した獲得関数には近似誤差が残ります。
また、数値計算中の乱数を固定すると候補間の比較が安定する場合がありますが、
それだけで最適候補の正確性が保証されるわけではありません。

analytic acquisition と MC acquisition の違いは「どちらが高度か」ではありません。
必要な期待値を閉形式で正確に計算できる設定なら analytic が有効であり、
複雑な joint posterior や utility を扱うために sampling が必要なら MC が自然です。

詳細は [Foundations](acquisition/01_foundations.md) で扱います。

## 4.6 Sequential、Batch、Noisy

1点ずつ評価する逐次 BO だけでなく、複数点を同時に評価する batch BO があります。
候補集合を

$$
X = \{x_1,\ldots,x_q\}
$$

とすると、q-acquisition は通常、各点の独立スコア上位を選ぶものではありません。
**候補集合の joint posterior と候補間相関を含めた価値**を評価します。

例えば2点がほぼ同じ情報しか与えないなら、単点スコアが両方高くても、
同時に選ぶ価値は低くなる場合があります。このため q-acquisition は
「1点用 acquisition を計算して上位 $`q`$ 点を取る」こととは異なります。

また、観測ノイズがある場合には、観測された最大値と潜在関数の真の最良値を区別する
必要があります。大きな正の観測ノイズによって偶然得られた値を $`f_{\mathrm{best}}`$
として固定すると、改善量の基準自体が歪む可能性があります。Noisy acquisition は、
このような latent function の不確実性も含めて意思決定するための考え方です。実行中で結果が未取得の pending points も、非同期 BO では候補選択へ
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

$$
f(x)=(f_1(x),\ldots,f_m(x))
$$

を同時に扱う場合、単一の最良値だけではなく Pareto dominance や Pareto front を
考えます。

Hypervolume-based acquisition では reference point に対する hypervolume improvement
を観測価値として使います。reference point は単なる描画範囲ではなく、
「どの領域の hypervolume を価値として数えるか」を定めるため、acquisition value に
直接影響します。EHVI / NEHVI はこの考え方に基づきます。一方、
NParEGO のように scalarization を利用する方法や、lookahead を組み合わせる方法も
あります。

詳細は [Multi-objective acquisition](acquisition/07_multiobjective.md) で扱います。

## 4.9 Constraints と Objective

実問題では、目的関数だけでなく

$$
g_j(x) \le 0
$$

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

例えば目的改善を $I(x)\ge0$、制約の可行性を
$A(x)=\{g_j(x)\le0\ \text{for all }j\}$ とすれば、制約付き期待改善は
$\mathbb E[I(x)\mathbf1_{A(x)}\mid\mathcal D_n]$ と定義できます。
これがEIと可行確率の積に分解できるのは、事後分布で改善量と可行性が独立な場合です。
さらに各制約の可行確率を掛けるには、制約間の条件付き独立性も必要です。
相関出力の場合は同時分布を用いる必要があります。

詳細は [Constrained acquisition](acquisition/08_constraints.md) で扱います。

ここでいう制約は、未知の応答を surrogate model で推定する **output / black-box constraint**
です。一方、組成和、変数間の大小関係、既知の線形設計則など、candidate の入力座標そのものに
課す制約は acquisition function ではなく acquisition optimization の責務です。

```text
output constraint:     g(x) <= 0  → model / acquisition composition
candidate constraint:  A x <= b   → acquisition optimizer
```

後者は既知の可行集合を定め、候補探索の段階で満たす必要があります。
この二種類を同じ制約として扱わず、確率的な応答制約と決定論的な入力制約を区別します。

## 4.10 Bayesian Optimization だけが目的ではない

posterior を使った逐次実験設計は、最適化だけに限定されません。

### Bayesian Optimization

目的は、良い入力、最適な入力、または最適目的値を効率よく見つけることです。

### Active Learning

目的は、関数や予測分布について効率よく学習することです。posterior variance、
integrated variance reduction、predictive information gain などが基準になります。

### Level-set Estimation

目的は、指定した target $`t`$ に対して

$$
f(x)=t
$$

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

候補 $(x,s)$ を忠実度 $s$ で観測したときの、目標忠実度に対する情報価値と
評価コスト $c(x,s)>0$ を組み合わせます。例えば、期待する意思決定の改善量を
$V(x,s)$ とすれば、$V(x,s)/c(x,s)$ は単位コスト当たりの価値です。
これは費用を考慮する方法の一つであり、限られた総予算の下での最適戦略と常に一致するわけではありません。
Knowledge Gradientと費用に応じた効用を組み合わせる方法は、その代表例です。

詳細は
[Multi-Fidelity and cost-aware acquisition](acquisition/11_multifidelity_cost_aware.md)
で扱います。

## 4.12 Acquisition Function とその最適化は別問題

獲得関数を定義することと、それを最大化することは別の問題です。

$$
\text{acquisition definition}
\neq
\text{acquisition optimization}
$$

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

## 4.14 Acquisition Function をどう選ぶか

初心者が最初からすべての acquisition function を使い分ける必要はありません。
まず「問題設定が何か」を切り分ける方が重要です。

| 問題設定 | 最初に確認する観点 |
| --- | --- |
| 単目的・逐次・低ノイズ | improvement / confidence bound |
| 単目的・batch | jointな q-acquisition が必要か |
| 観測ノイズが大きい | noisy formulation が必要か |
| 多目的 | Pareto / hypervolume / scalarization のどれを使うか |
| 未知制約あり | feasibility をどう acquisition に組み込むか |
| 評価コストが異なる | information value と cost の関係 |
| 最適化ではなく学習 | Active Learning / Level-set の目的を定義する |

これは手法の優劣表ではありません。同じposteriorを使っていても、
**何を意思決定上の価値と定義するか**によって適切な acquisition family が変わります。

## 4.16 詳細理論への入口

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

詳細章では、EI・PI・UCBの導出から、KGによる先読み、多目的のEHVI・NEHVIまで、
それぞれの効用・統計的仮定・近似方法を確認できます。

## 4.17 この章で覚えておくこと

Acquisition Function は、surrogate model の posterior を

**次に何を観測する価値があるか**

という意思決定へ変換する層です。最も重要なのは、acquisition function が
「真の目的関数」でも「posterior mean」でもなく、**限られた評価予算を次にどこへ
使うかを決めるための utility** だと理解することです。

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

## 参考文献

1. Jones, D. R., Schonlau, M., and Welch, W. J. (1998).
   Efficient global optimization of expensive black-box functions.
   *Journal of Global Optimization*, 13, 455-492.
2. Frazier, P. I. (2018).
   A tutorial on Bayesian optimization.
   arXiv:1807.02811.
3. Wilson, J., Hutter, F., and Deisenroth, M. (2018).
   Maximizing acquisition functions for Bayesian optimization.
   *Advances in Neural Information Processing Systems 31*.
4. Balandat, M. et al. (2020).
   BoTorch: A framework for efficient Monte-Carlo Bayesian optimization.
   *Advances in Neural Information Processing Systems 33*.
5. [Acquisition Function Theory](acquisition/README.md)

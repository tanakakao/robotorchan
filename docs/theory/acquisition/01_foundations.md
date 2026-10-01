# 1. Foundations

## 1.1 まず全体の流れ

~~~text
observed data -> surrogate -> posterior -> objective / utility
              -> acquisition value -> candidate optimization -> next candidate
~~~

surrogateは未知応答の予測、acquisitionは次の評価の価値付け、candidate optimizerは
そのacquisitionの最大化を担当します。この3層を最初から分けて考えます。

## 1.2 獲得関数を意思決定として見る

獲得関数は surrogate model の予測器そのものではなく、posterior distribution を「次の観測の価値」へ変換する意思決定基準です。

観測データを

$
\mathcal D_n=\{(x_i,y_i)\}_{i=1}^n
$

とし、潜在関数の posterior を

$
p(f\mid\mathcal D_n)
$

とします。逐次選択では一般に

$
x_{n+1}\in\arg\max_{x\in\mathcal X}\alpha(x;\mathcal D_n)
$

として候補を選びます。

重要なのは、$`\alpha`$ が「予測値」ではなく、posterior に対して定義した **utility の期待値または代理量**だという点です。

## 1.3 Posterior と utility

候補集合 $`X`$ の未知出力を \(f(X)\) とし、その結果から得られる効用を \(U(f(X))\) とすれば、多くの獲得関数は概念的に

$
\alpha(X)
=
\mathbb E_{p(f(X)\mid\mathcal D_n)}
[U(f(X))]
$

と捉えられます。

ただし、すべての手法がこの単純な形に直接書けるわけではありません。
Information-theoretic acquisitionではentropyやmutual information、lookaheadでは
将来のposteriorと意思決定価値を扱います。

utilityは目的関数そのものとも限りません。EIではimprovementがutilityになり、
Active Learningでは予測を学ぶ価値がutilityになります。

## 1.4 Exploration と exploitation

単一出力 Gaussian posterior

$
f(x)\mid\mathcal D_n
\sim
\mathcal N(\mu_n(x),\sigma_n^2(x))
$

を考えると、\(\mu_n(x)\) は現在の予測、\(\sigma_n(x)\) は posterior uncertainty を表します。

- exploitation は現在良いと予測される領域を重視します。
- exploration は不確実で、追加観測により知識が変わり得る領域を重視します。

ただし、この二分法は直感であって、すべての獲得関数を「平均項 + 分散項」として
理解すべきではありません。EIはimprovement、MESはoptimum valueの情報量、KGは将来の
意思決定価値を直接定義します。

## 1.5 Analytic acquisition

posteriorとutilityの組合せによって期待値が閉形式で計算できる場合、analytic acquisitionを
構成できます。analyticは手法の目的ではなく、閉形式で評価できるという計算上の性質です。

Gaussian posteriorに対するq=1のEIやPIは代表例です。analytic formは高速ですが、
閉形式が成立するposterior、objective、batch構造に制約があります。

## 1.6 Monte Carlo acquisition

閉形式が難しい場合、posterior sample

$
f^{(s)}(X)\sim p(f(X)\mid\mathcal D_n),
\quad s=1,\ldots,S
$

を使って

$
\alpha(X)
\approx
\frac{1}{S}
\sum_{s=1}^{S}U(f^{(s)}(X))
$

と近似できます。

Monte Carlo（MC）formulation は、

- q > 1 の joint candidate selection
- multi-output posterior
- nonlinear objective
- constraints
- noisy observations

などへ拡張しやすいことが利点です。

MC errorはsample数に依存します。最適化中にacquisition surfaceが不必要に揺れないよう、
QMC samplingやbase samplesの扱いも実装上重要になります。

### MC samplingとjoint posterior

batch utilityでは、同じposterior sample内でcandidate間のjoint structureを保つことが重要です。
各candidateのmarginalから独立sampleを作るだけでは、本来のjoint utilityと異なる場合があります。

したがってsample可能であることと、acquisitionが要求するjoint sampling semanticsを満たすことを
区別します。

## 1.7 Objective と posterior transform

modelがmulti-outputでも、model outputとacquisitionが使う意思決定量は同じとは限りません。

posterior sample \(f(X)\) に対して scalar objective

$
h(f(X))
$

を定義し、その後 utility を評価する場合があります。

$
\text{posterior}
\rightarrow
\text{objective / transform}
\rightarrow
\text{utility}
\rightarrow
\text{acquisition value}
$

したがって、model output、objective、acquisition functionは別の概念です。
Multi-objective BOのようにobjective vector自体を扱う場合は、この単純なscalarizationとは
異なる基準も利用します。

sample-level objectiveとPosteriorTransformも同一ではありません。前者はposterior sampleを
utility計算用の量へ写せますが、後者はposterior representation自体を変換します。

## 1.8 Acquisition definition と optimization

獲得関数の定義と、その最大化は別問題です。

$
X^*\in\arg\max_{X\in\mathcal X^q}\alpha(X)
$

を数値的に解く必要があります。

連続空間ではgradient-based optimizationを使える場合があります。一方、categorical、integer、
hierarchical、discrete fidelity、制約付き空間では、acquisition optimization側に専用の
探索方法が必要です。

良いacquisition criterionが、必ずしも最適化しやすいacquisition surfaceを持つとは限りません。

また、既知のboundsやcandidate Xに対するlinear / nonlinear constraintはoptimizer側の責務です。
未知のfeasibility outcomeをmodelingするblack-box constraintとは区別します。

## 1.9 Surrogate uncertainty の品質

獲得関数は posterior を信頼して動きます。

uncertaintyを過小評価すれば探索不足、過大評価すれば過剰探索につながり得ます。
kernel、noise model、task structure、input transform、approximate inferenceなどは
acquisitionとソフトウェア上別の責務でも、意思決定品質には直接影響します。

### posterior uncertaintyの意味もmodel依存

Exact GPのposterior variance、ensemble disagreement、distributional regressionのpredictive
varianceを同じ統計量だとは仮定しません。batchやinformation-based acquisitionでは、
marginal momentsだけでなくcandidate間のjoint structureが必要になる場合もあります。

## 1.10 BoTorch / robotorchan との対応

BoTorchはanalytic / MC acquisition、objective、posterior transform、sampler、
acquisition optimizerを分離した設計を採用しています。

robotorchanのcapability metadataではposterior requirementをmarginal moments、joint Gaussian、
posterior samplesに分けます。さらにq、multi-output、ensemble、fantasization等が重なります。

robotorchanもBoTorch-firstとし、BoTorch native acquisitionが目的を満たす場合は
ローカルwrapperを作りません。現在の対応状況は
[Acquisition integration status](../../optimization/acquisition-integration.md) を参照してください。

## 1.11 この章で覚えておくこと

- acquisitionはposterior predictionそのものではなくdecision valueである
- utilityは目的関数そのものとは限らない
- analytic / MCは主としてacquisition valueの評価方法に関する分類である
- q-acquisitionではcandidate集合のjoint posterior structureが重要になる
- objectiveとPosteriorTransformは同じ責務ではない
- acquisition definitionとcandidate optimizationを分ける
- black-box outcome constraintとknown candidate constraintを分ける
- posterior interfaceが同じでもuncertaintyの統計的意味が同じとは限らない

## 1.12 次に読む章

標準的な単目的BOの基準として、次は [Improvement](02_improvement.md) と
[Confidence Bound](03_confidence_bound.md) を扱います。batch、noise、pending pointsは
[Batch / Noisy](04_batch_noisy.md) で整理します。

## 参考文献

1. Frazier, P. I. (2018). A Tutorial on Bayesian Optimization. arXiv:1807.02811.
2. Balandat, M. et al. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
   *Advances in Neural Information Processing Systems 33*.
3. Wilson, J., Hutter, F., and Deisenroth, M. (2018). Maximizing acquisition functions for Bayesian
   optimization. *Advances in Neural Information Processing Systems 31*.

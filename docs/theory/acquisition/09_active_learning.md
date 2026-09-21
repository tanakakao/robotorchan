# 9. Active Learning Acquisition

## 9.1 最適化ではなく学習を目的にする

Bayesian Optimization（BO）と Active Learning（AL）は、どちらも surrogate posterior を使って次の観測点を選べますが、目的が異なります。

```text
Bayesian Optimization
    良い入力・最適値を効率よく見つける

Active Learning
    未知関数や prediction を効率よく学習する
```

BO では objective optimum に関係する領域が重要です。一方、AL では prediction quality、posterior uncertainty、target distribution 上の情報量などが観測価値になります。

したがって、AL acquisition を「改善量を使わない BO acquisition」と理解するのは不十分です。

## 9.2 Pool-based と continuous-domain AL

候補集合が有限 pool

\[
\mathcal X_{\mathrm{pool}}
=
\{x^{(1)},\ldots,x^{(N)}\}
\]

として与えられる場合、acquisition score を pool 上で評価して選択できます。

一方、連続空間 \(\mathcal X\) では

\[
x^*
\in
\arg\max_{x\in\mathcal X}\alpha_{AL}(x)
\]

として acquisition optimization が必要です。

理論上同じ acquisition criterion でも、finite-pool selection と continuous optimization では計算方法と重複候補処理が異なります。

## 9.3 Posterior Variance Sampling

最も直接的な uncertainty sampling は posterior variance

\[
\alpha_{\mathrm{Var}}(x)
=
\operatorname{Var}[f(x)\mid\mathcal D_n]
=
\sigma_n^2(x)
\]

を最大化する方法です。

posterior variance が大きい点は、現在の surrogate がその latent function value を十分に知らない領域です。

この基準は objective value の大小を使わないため、最適値探索ではなく function learning に自然です。

## 9.4 Posterior Standard Deviation

posterior standard deviation を使えば

\[
\alpha_{\mathrm{Std}}(x)
=
\sigma_n(x)
\]

です。

variance と standard deviation は q=1 の単純 ranking では単調変換なので、同じ maximizer を持ちます。

ただし acquisition value の scale は異なるため、

- 他の score と加算する
- penalty と組み合わせる
- threshold を設ける
- optimization landscape の scaling が影響する

場合には完全に同じ実装的意味になるとは限りません。

## 9.5 Aleatoric noise と epistemic uncertainty

AL で重要なのは、uncertainty の種類を区別することです。

観測モデルを

\[
y(x)=f(x)+\epsilon
\]

とすると、predictive variance は概念的に

\[
\operatorname{Var}[y(x)\mid\mathcal D_n]
=
\operatorname{Var}[f(x)\mid\mathcal D_n]
+
\operatorname{Var}[\epsilon]
\]

と分解できます。

追加データで減らしたいのは主に latent function に関する epistemic uncertainty です。

irreducible observation noise が大きい場所を単純に predictive variance が高いという理由だけで選ぶと、何度観測しても十分には減らない aleatoric uncertainty を追い続ける可能性があります。

したがって「どの posterior variance を acquisition が使っているか」を確認する必要があります。

## 9.6 Pointwise uncertainty の限界

PosteriorVariance はその候補自身の uncertainty を評価します。

しかし候補 \(x\) を観測すると、GP の covariance structure により他の点 \(x'\) の uncertainty も変化します。

したがって、

```text
その点がどれだけ不確実か
```

と

```text
その点を観測すると領域全体の uncertainty がどれだけ減るか
```

は別の criterion です。

後者を扱うのが integrated variance reduction の考え方です。

## 9.7 Integrated Posterior Variance

関心領域または target distribution \(p_T(x)) に対する integrated posterior variance を

\[
V_n
=
\int_{\mathcal X}
\sigma_n^2(x)
p_T(x)\,dx
\]

とします。

候補 \(x_c\) を観測した後の posterior variance を \(\sigma_{n+1}^2\) とすれば、expected integrated variance reduction は

\[
\Delta V(x_c)
=
V_n
-
\mathbb E_{Y_{x_c}}
\left[
\int
\sigma_{n+1}^2(x)
p_T(x)\,dx
\right]
\]

です。

Gaussian models の条件によっては posterior covariance update が観測値そのものに依存しないため、この expectation を簡略化できる場合があります。

## 9.8 Target distribution の意味

AL の目的は「探索空間全体を均等に学ぶ」とは限りません。

実運用で prediction する入力分布を \(p_T(x)\) とすれば、target distribution 上で重要な領域へ acquisition を集中できます。

例えば、

- 製造条件として実際に出現する領域
- 将来の deployment distribution
- 特定の ROI
- finite target set

を target として扱えます。

これは BO の optimum-oriented utility とは異なる設計軸です。

## 9.9 Negative Integrated Posterior Variance

BoTorch の qNegativeIntegratedPosteriorVariance（qNIPV）は、candidate batch を観測した後の integrated posterior variance を基準に candidate set を評価する考え方です。

最小化したい posterior variance を負号によって最大化問題として扱います。

概念的には

\[
\alpha_{\mathrm{NIPV}}(X)
=
-
\int
\operatorname{Var}
[
f(x)
\mid
\mathcal D_n, X
]
p_T(x)\,dx
\]

です。

実際には integration points による数値近似を使います。

qNIPV の重要な点は、単に q 個の高分散点を選ぶのではなく、candidate set が target region 全体の uncertainty をどのように減らすかを評価することです。

## 9.10 Batch Active Learning

q > 1 の AL でも redundancy が問題になります。

posterior variance の高い点を独立に上位 q 点選ぶと、互いに強く相関した近接点が選ばれる可能性があります。

joint batch criterion は、

\[
X=(x_1,\ldots,x_q)
\]

をまとめて条件付けした後の uncertainty reduction を評価することで、候補間の情報重複を考慮できます。

これは BO の q-acquisition と同じく、「pointwise score 上位 q 点」と「joint q acquisition」が異なることを示します。

## 9.11 Predictive Information Gain

variance reduction は second-order uncertainty を基準にします。

より一般には、候補観測 \(Y_x\) と target prediction \(Y_T\) の mutual information

\[
I(Y_x;Y_T\mid\mathcal D_n)
\]

を acquisition value として使えます。

これは

```text
candidate を観測すると
target prediction について
どれだけ情報を得られるか
```

を直接評価します。

Gaussian setting では covariance structure と entropy の関係から、variance reduction と information gain は密接に関係しますが、概念としては同一ではありません。

## 9.12 EPIG

Expected Predictive Information Gain（EPIG）は、target inputs における predictions と candidate observation の information gain を target distribution について平均します。

target input を \(x_T\sim p_T\) とすれば、概念的に

\[
\operatorname{EPIG}(x)
=
\mathbb E_{x_T\sim p_T}
\left[
I(Y_x;Y_{x_T}\mid\mathcal D_n)
\right]
\]

です。

finite target set

\[
T=\{x_T^{(1)},\ldots,x_T^{(M)}\}
\]

と weights \(w_j\) を使えば、

\[
\operatorname{EPIG}(x)
\approx
\sum_{j=1}^{M}
w_j
I(Y_x;Y_{x_T^{(j)}}\mid\mathcal D_n)
\]

として評価できます。

## 9.13 EPIG と Entropy Sampling の違い

candidate 自身の predictive entropy

\[
H(Y_x\mid\mathcal D_n)
\]

を最大化するだけでは、その候補が target predictions に有益かは分かりません。

EPIG は candidate uncertainty ではなく、candidate と target prediction の dependence を評価します。

したがって、

```text
Entropy / Variance Sampling
    candidate 自身がどれだけ不確実か

EPIG
    candidate を知ると target prediction をどれだけ学べるか
```

という違いがあります。

## 9.14 EPIG と MES の違い

どちらも mutual information を使えますが、情報対象が異なります。

```text
MES
    optimum value f* が情報対象
    → Bayesian Optimization

EPIG
    target prediction が情報対象
    → Active Learning
```

したがって information-theoretic acquisition を分類するときは、式に entropy が出てくるかではなく「何について情報を得るのか」を見る必要があります。

## 9.15 Regression と Classification

AL は regression だけでなく classification でも重要です。

classification では、

- predictive entropy
- BALD
- margin uncertainty
- probability variance
- expected information gain

などが利用されます。

ただし classification posterior は Gaussian regression posterior と同じ構造ではありません。latent function uncertainty、class probability uncertainty、label entropy を区別する必要があります。

本章は robotorchan の現在の regression acquisition theory を中心に扱い、classification-specific acquisition を同じ数式へ無理に統合しません。

## 9.16 Distribution shift

target distribution \(p_T\) が deployment distribution と一致しない場合、target-aware AL は誤った領域を重視する可能性があります。

また pool distribution と target distribution が異なる場合、

```text
どこから観測できるか
```

と

```text
どこで予測精度が必要か
```

を分離する必要があります。

これは covariate shift を伴う experimental design において重要です。

## 9.17 Cost-aware Active Learning

観測点ごとに評価コストが異なる場合、information gain だけでなく cost も考慮できます。

例えば概念的には

\[
\frac{\text{expected information gain}}
{\text{evaluation cost}}
\]

のような utility が考えられます。

ただし単純な ratio が常に最適とは限りません。cost-aware decision は budget、terminal objective、fidelity structure に依存します。

Multi-Fidelity BO の value-of-information / cost の考え方とは関連しますが、AL では target prediction learning が最終目的です。

## 9.18 Stopping criterion

AL では optimum discovery 以外の stopping rule が自然です。

例えば、

- maximum posterior variance が閾値以下
- integrated posterior variance が十分小さい
- expected information gain が小さい
- target-set predictive uncertainty が要求値以下
- labeling / experiment budget を使い切った

などがあります。

acquisition criterion と stopping criterion は別の責務ですが、同じ uncertainty objective に基づいて設計できる場合があります。

## 9.19 BoTorch / robotorchan との対応

BoTorch は qNegativeIntegratedPosteriorVariance など experimental-design 向け acquisition を提供しています。

robotorchan は regression AL 向けに PosteriorVariance、PosteriorStd、ExpectedPredictiveInformationGain を提供し、BoTorch native で十分な手法は再実装しません。

現在の robotorchan-specific acquisition には q、structured output、ensemble posterior などに明示的な適用範囲があります。Theory 上の一般形と現在の実装範囲を混同せず、詳細は次を参照してください。

- [Regression Active Learning guide](../../optimization/regression_active_learning.md)
- [EPIG guide](../../optimization/epig.md)
- [Acquisition integration status](../../optimization/acquisition-status.md)

## 9.20 まとめ

Active Learning acquisition は「どこが最適か」ではなく「何を効率よく学びたいか」から設計します。

```text
PosteriorVariance / PosteriorStd
    candidate 自身の uncertainty

Integrated variance / qNIPV
    target region 全体の uncertainty reduction

EPIG
    target prediction についての information gain
```

target distribution を明示すると、単なる空間充填ではなく deployment に必要な prediction quality を直接意識できます。

次章 [Level-set](10_level_set.md) では、関数全体ではなく指定した response level の境界を学ぶ acquisition を扱います。

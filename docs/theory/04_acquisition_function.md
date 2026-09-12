# 4. Acquisition Function

## 4.1 Acquisition Function とは

ベイズ最適化では、Gaussian Process（GP）などの surrogate model から得られる予測分布を使って、**次にどの点を評価するか**を決めます。

この意思決定を数値化する関数が acquisition function（獲得関数）です。

現在までの観測データを

\[
\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n
\]

とし、候補点 `x` に対する獲得関数を

\[
\alpha(x; \mathcal{D}_n)
\]

と書くと、次の評価点は典型的に

\[
x_{n+1}
=
\operatorname*{arg\,max}_{x \in \mathcal{X}}
\alpha(x; \mathcal{D}_n)
\]

として選びます。

つまり、ベイズ最適化は

```text
surrogate model
    ↓
posterior mean / uncertainty
    ↓
acquisition function
    ↓
次に評価する候補点
```

という流れで動きます。

## 4.2 なぜ目的関数の予測値を直接最大化しないのか

単純に posterior mean

\[
\mu_n(x)
\]

だけを最大化すると、現在のモデルがすでに高いと考えている場所ばかりを選びやすくなります。

しかし、まだ観測していない領域には大きな不確実性

\[
\sigma_n(x)
\]

が残っています。

その領域を調べれば、現在の予想を大きく上回る点が見つかる可能性があります。

したがって獲得関数は、多くの場合

- **Exploitation（活用）**: 予測平均が高い点を選ぶ
- **Exploration（探索）**: 不確実性が高い点を調べる

のバランスを取ります。

## 4.3 Posterior distribution が入力になる

単一出力GPでは、候補点 `x` に対して

\[
f(x) \mid \mathcal{D}_n
\sim
\mathcal{N}(\mu_n(x), \sigma_n^2(x))
\]

という posterior distribution が得られます。

獲得関数は、この分布を異なる基準で評価していると考えると理解しやすくなります。

例えば、

```text
EI  : どれだけ改善できそうか
PI  : 改善する確率はどれくらいか
UCB : 高い値になる可能性をどこまで楽観視するか
KG  : その観測によって将来の意思決定がどれだけ改善するか
```

という違いです。

## 4.4 Expected Improvement（EI）

現在の最良値を

\[
f_{\mathrm{best}}
\]

とします。

最大化問題では、候補点 `x` における improvement を

\[
I(x)
=
\max(f(x)-f_{\mathrm{best}}, 0)
\]

と定義できます。

Expected Improvement（EI）は、この improvement の期待値です。

\[
\operatorname{EI}(x)
=
\mathbb{E}[I(x)]
\]

GP posterior が正規分布である場合、

\[
z(x)
=
\frac{\mu_n(x)-f_{\mathrm{best}}}{\sigma_n(x)}
\]

とすると、

\[
\operatorname{EI}(x)
=
(\mu_n(x)-f_{\mathrm{best}})\Phi(z)
+
\sigma_n(x)\phi(z)
\]

と書けます。

ここで

- `\Phi` : 標準正規分布の累積分布関数
- `\phi` : 標準正規分布の確率密度関数

です。

### EI の直感

EIには2つの項があります。

\[
(\mu-f_{\mathrm{best}})\Phi(z)
\]

は主に exploitation を表し、

\[
\sigma\phi(z)
\]

は uncertainty による exploration を表します。

したがってEIは、探索と活用のバランスを比較的自然に取る獲得関数です。

## 4.5 LogEI

EIは非常に代表的な獲得関数ですが、改善確率や改善量が極端に小さい領域では、数値的に値がゼロへ潰れやすくなります。

そのため現在のBoTorchでは、EIそのものよりも、数値的に安定な logarithmic formulation を使う `LogExpectedImprovement` や `qLogExpectedImprovement` が推奨されます。

重要なのは、LogEIがまったく別の探索基準というより、**EIの最適化を数値的に安定化した実装**として理解することです。

実務上は、標準的な単目的BOではまず LogEI 系を候補にするのが自然です。

## 4.6 Probability of Improvement（PI）

Probability of Improvement（PI）は、候補点が現在の最良値を超える確率を評価します。

\[
\operatorname{PI}(x)
=
P(f(x) > f_{\mathrm{best}})
\]

正規posteriorなら

\[
\operatorname{PI}(x)
=
\Phi\left(
\frac{\mu_n(x)-f_{\mathrm{best}}}{\sigma_n(x)}
\right)
\]

です。

PIは「どれだけ改善するか」を考えず、「改善する確率」だけを見ます。

例えば、

```text
候補A: 0.1だけ改善する確率が90%
候補B: 5.0改善する確率が45%
```

であればPIはAを好みやすい一方、EIはBの大きな改善量も評価します。

そのためPIは直感的ですが、局所的な改善を繰り返して早く収束しすぎる場合があります。

## 4.7 Upper Confidence Bound（UCB）

UCBは、posterior mean と posterior standard deviation を直接組み合わせます。

最大化問題では典型的に

\[
\operatorname{UCB}(x)
=
\mu_n(x)
+
\sqrt{\beta}\,\sigma_n(x)
\]

と書きます。

ここで `\beta` は探索の強さを調整するパラメータです。

### 小さい beta

\[
\beta \downarrow
\]

では posterior mean の影響が強くなり、exploitation 寄りになります。

### 大きい beta

\[
\beta \uparrow
\]

では uncertainty の寄与が大きくなり、exploration 寄りになります。

UCBの利点は、探索・活用のバランスを非常に明示的に制御できる点です。

一方で `beta` の選び方が探索挙動に直接影響します。

## 4.8 EI・PI・UCB の違い

代表的な3つを整理すると次のようになります。

| 獲得関数 | 主に評価するもの | 特徴 |
|---|---|---|
| EI / LogEI | 期待改善量 | 改善確率と改善量を両方考える |
| PI | 改善確率 | 単純だが小さな改善へ偏りやすい |
| UCB | 平均 + 不確実性 | 探索強度を `beta` で直接制御できる |

単目的の標準的なBOでは、まずLogEIを基準とし、探索強度を意図的に制御したい場合にUCBを比較する、という使い方が分かりやすいです。

## 4.9 Knowledge Gradient（KG）

EIやUCBは、基本的に「その候補点自身がどれだけ良さそうか」を評価します。

Knowledge Gradient（KG）はより一歩進めて、

**その候補点を観測した結果、次の意思決定がどれだけ改善するか**

を評価します。

概念的には

\[
\operatorname{KG}(x)
=
\mathbb{E}
\left[
V(\mathcal{D}_{n+1})
-
V(\mathcal{D}_n)
\right]
\]

のような value of information として捉えられます。

ここで `V` は、現在の情報で最終的に選べる解の価値です。

KGは情報価値を明示的に扱えるため、

- 観測コストが高い
- 残り試行回数が重要
- Multi-Fidelity
- lookahead 的な意思決定

などで特に意味を持ちます。

ただしEIやUCBより計算コストが高くなりやすい点に注意が必要です。

## 4.10 Analytic acquisition と Monte Carlo acquisition

獲得関数には、大きく

- analytic acquisition
- Monte Carlo（MC）acquisition

があります。

### Analytic

posterior distribution に対して期待値を閉形式で計算できる場合に使います。

典型的には

```text
q = 1
単一出力
比較的単純な objective
```

で利用しやすくなります。

### Monte Carlo

posteriorからサンプル

\[
f^{(s)}(X), \quad s=1,\ldots,S
\]

を生成し、

\[
\alpha(X)
\approx
\frac{1}{S}
\sum_{s=1}^{S}
U(f^{(s)}(X))
\]

のように期待効用を近似します。

MC acquisition は

- batch BO
- multi-output
- nonlinear objective
- constraints
- noisy observation

などへ拡張しやすいことが大きな利点です。

BoTorchがMC acquisitionを重視している理由の1つは、この柔軟性にあります。

## 4.11 q-acquisition と Batch Bayesian Optimization

1回の実験サイクルで `q` 点を同時に評価したい場合、候補集合

\[
X = \{x_1,\ldots,x_q\}
\]

をまとめて評価します。

例えば qEI は、単純に1点ずつEIが高い点を選ぶのではなく、**q点を同時に選んだときの共同改善量**を考えます。

これは重要です。

単純に上位q点を取ると、互いに非常に近い候補が選ばれる可能性があります。

q-acquisitionでは posterior correlation を考慮するため、候補集合としての価値を評価できます。

代表例として

- `qLogExpectedImprovement`
- `qUpperConfidenceBound`
- `qKnowledgeGradient`

などがあります。

## 4.12 観測ノイズと Noisy Expected Improvement

実験値にノイズがある場合、観測された最大値

\[
\max_i y_i
\]

が真の最良値とは限りません。

その場合、単純なEIよりも、潜在的な真の関数値に関する不確実性を考慮する Noisy Expected Improvement（NEI）が適します。

BoTorchでは、数値安定性の観点から `qLogNoisyExpectedImprovement` のようなlog formulationを優先するのが実務的です。

特に

```text
実験ばらつきが大きい
測定誤差が無視できない
同一点を測定しても値が揺れる
```

ような状況では、noise-aware acquisitionを選ぶ意味が大きくなります。

## 4.13 Pending points と非同期BO

実験や計算を並列実行すると、すでに投入済みだが結果がまだ返っていない点が存在します。

これを pending points と考えます。

pending pointsを無視すると、獲得関数が同じ領域へ追加候補を出し、実験資源を重複投入することがあります。

非同期BOでは、未完了点を `X_pending` として acquisition に反映し、

```text
すでに走っている実験
    ↓
その情報が得られる前提も考慮
    ↓
次の候補を選ぶ
```

という処理を行います。

Batch BOを実運用へ持ち込む際には重要な考え方です。

## 4.14 Multi-Objective Bayesian Optimization

複数目的

\[
f(x)=
(f_1(x),\ldots,f_m(x))
\]

を同時に最適化する場合、単一の最良値ではなく Pareto front を改善することが目標になります。

このとき代表的な指標が hypervolume です。

現在のPareto集合が占めるhypervolumeを `HV` とすると、候補を追加したときの

\[
\Delta HV
\]

を改善量として扱えます。

## 4.15 EHVI

Expected Hypervolume Improvement（EHVI）は、候補点を評価した結果として得られる hypervolume improvement の期待値です。

\[
\operatorname{EHVI}(x)
=
\mathbb{E}[\Delta HV(x)]
\]

単目的のEIに対応するmulti-objective版と考えると理解しやすくなります。

複数点を同時に選ぶ場合は qEHVI 系を使います。

現在のBoTorchでは、multi-objectiveでも数値安定性を考慮したlog formulationである `qLogExpectedHypervolumeImprovement` が重要な選択肢です。

## 4.16 NEHVI

観測ノイズがあるmulti-objective BOでは、Noisy Expected Hypervolume Improvement（NEHVI）が重要になります。

単なる観測値のPareto frontではなく、潜在関数に対する不確実性を考慮しながらhypervolume improvementを評価します。

実験データを扱う場合、multi-objectiveかつnoiseありという条件は珍しくないため、NEHVI系は実務上非常に重要です。

BoTorchでは `qLogNoisyExpectedHypervolumeImprovement` の利用が代表的です。

## 4.17 Pareto front と Reference Point

Hypervolume-based acquisitionでは reference point が必要です。

reference pointは、目的空間で「これより悪い領域」を定める基準です。

最大化問題なら通常、関心のあるPareto frontより十分悪い点を設定します。

reference pointが不適切だと、hypervolume improvementの尺度そのものが変わるため、候補選択へ大きく影響します。

したがってEHVI / NEHVIでは、獲得関数名だけでなく reference point の設計も重要です。

## 4.18 制約付き Bayesian Optimization

実問題では、目的関数だけでなく制約

\[
g_j(x) \le 0
\]

を満たす必要があります。

制約付きBOでは、概念的には

\[
\alpha_{\mathrm{constrained}}(x)
\approx
\alpha_{\mathrm{objective}}(x)
\times
P(\text{feasible at }x)
\]

のように、改善価値と実行可能性を組み合わせます。

例えば目的値が非常に高そうでも、制約違反確率が高い点は優先度が下がります。

MC acquisitionでは、posterior sampleごとに constraint indicator や滑らかな feasibility weighting を適用できるため、複雑な制約へ拡張しやすくなります。

## 4.19 Acquisition Function と Objective は別物

BoTorchを扱う際には、acquisition function と objective を分けて考えることが重要です。

モデルがmulti-outputであっても、最適化したい量が

\[
h(f_1(x),f_2(x),\ldots)
\]

のようなscalar objectiveなら、posterior sampleをobjectiveでscalar化してから acquisition value を計算できます。

つまり

```text
Model posterior
    ↓
Objective
    ↓
Utility / Improvement
    ↓
Acquisition value
```

という分離があります。

この分離により、同じmodelを使ったまま最適化目的だけを変更できます。

## 4.20 Acquisition Function とその最適化も別物

もう1つ重要なのは、

- acquisition function を定義すること
- acquisition function を最大化すること

は別問題だという点です。

連続変数なら勾配ベース最適化が利用できますが、

- categorical variable
- integer variable
- hierarchical variable
- equality / inequality constraint

などがあると acquisition optimization 自体が難しくなります。

したがって

\[
\text{良い acquisition function}
\neq
\text{簡単に最適化できる acquisition function}
\]

です。

次章のMixed Variablesでは、この「acquisition optimization側の難しさ」も重要になります。

## 4.21 Acquisition Function と surrogate model の相互作用

獲得関数はmodel posteriorを信頼して動きます。

したがって、surrogate modelが不適切なら、獲得関数だけを高度化しても良い候補は得られません。

例えば

```text
モデルが不確実性を過小評価
    ↓
探索が弱くなる

モデルが不確実性を過大評価
    ↓
不要な探索が増える
```

ということが起こります。

Kernel、noise model、task structure、input transformなどの設計は、acquisition functionと独立ではありません。

## 4.22 どの Acquisition Function を選ぶか

実務上の大まかな目安は次の通りです。

| 状況 | 最初に検討しやすい獲得関数 |
|---|---|
| 単目的・低ノイズ・逐次 | LogEI |
| 単目的・batch | qLogEI |
| 単目的・観測ノイズあり | qLogNEI |
| 探索強度を明示制御したい | UCB / qUCB |
| 情報価値を重視 | KG / qKG |
| 多目的・低ノイズ | qLogEHVI |
| 多目的・観測ノイズあり | qLogNEHVI |
| 制約あり | constraint-aware MC acquisition |

これは絶対的な規則ではありません。

問題の次元、観測ノイズ、batch size、制約、残り評価回数、surrogate modelの質などで適切な選択は変わります。

## 4.23 BoTorch / robotorchan との対応

robotorchanはBoTorch-nativeな設計を基本としているため、獲得関数についてもBoTorchのオブジェクトと組み合わせる考え方が基本になります。

例えば標準GPを学習した後、

```python
from botorch.acquisition import LogExpectedImprovement

best_f = model.raw_train_Y.max()
acqf = LogExpectedImprovement(
    model=model,
    best_f=best_f,
)
```

のようにmodel posteriorを獲得関数へ渡します。

候補生成は概念的に

```python
from botorch.optim import optimize_acqf

candidate, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=10,
    raw_samples=256,
)
```

のように acquisition optimization として実行します。

robotorchan側で今後 acquisition wrapper や optimizer wrapper を追加する場合も、この責務分離を崩さないことが重要です。

## 4.24 数値安定性を軽視しない

獲得関数は最終的に最適化対象になるため、値そのものだけでなく勾配の品質が重要です。

EI系では、改善確率が非常に小さい領域で acquisition value や gradient が数値的に潰れることがあります。

そのためBoTorch 0.18.1では、legacyなEI / qEI / qNEHVIをそのまま使うより、LogEI / qLogEI / qLogNEHVIなどのlog formulationを優先する設計が推奨されています。

「理論式として正しい」ことと「数値最適化で安定している」ことは別問題です。

## 4.25 まとめ

Acquisition Functionは、surrogate modelのposteriorを

**次にどこを評価するかという意思決定へ変換する層**

です。

重要な整理は次の通りです。

```text
EI / LogEI
  └─ 期待改善量

PI
  └─ 改善確率

UCB
  └─ posterior mean + uncertainty

KG
  └─ 観測による将来の意思決定価値

q-acquisition
  └─ 複数候補を共同で評価

NEI
  └─ observation noiseを考慮

EHVI / NEHVI
  └─ Pareto front / hypervolumeを改善

Constrained acquisition
  └─ objective valueとfeasibilityを同時に評価
```

そして、獲得関数の性能はsurrogate modelのposterior qualityに依存します。

したがってベイズ最適化では、

```text
Model
  ↓
Posterior
  ↓
Objective
  ↓
Acquisition Function
  ↓
Acquisition Optimization
  ↓
Candidate
```

という責務の分離を理解しておくことが重要です。

次章では、連続変数だけでなくカテゴリ変数を含む探索空間を扱うための [Mixed Variables](05_mixed_variables.md) を説明します。

## 参考

- BoTorch v0.18.1 Getting Started
- BoTorch v0.18.1 Multi-Objective Bayesian Optimization
- Ament et al., *Unexpected Improvements to Expected Improvement for Bayesian Optimization*, 2023

# 3. Confidence-bound Acquisition

[基礎章](01_foundations.md)の正規事後分布を前提とします。
標準偏差の係数を $\sqrt\beta$ と書き、点ごとの分位点と、全候補・全時刻を同時に覆う保証を分けて考えます。

## 3.1 UCBを一言で理解する

~~~text
UCB = 現在よさそうな候補 + まだ分からない候補へのoptimism bonus
~~~

EIがimprovementをutilityにするのに対し、UCBはposterior locationとuncertaintyを
直接組み合わせて候補を順位付けします。

## 3.2 平均と不確実性を直接組み合わせる

Confidence-bound acquisition は、posterior mean と posterior uncertainty を明示的に組み合わせます。

最大化問題の代表的な Upper Confidence Bound（UCB）は

$$
\operatorname{UCB}(x)
=
\mu_n(x)+\sqrt{\beta}\,\sigma_n(x)
$$

です。

ここで $`\mu_n(x)`$ は posterior mean、$`\sigma_n(x)`$ は posterior standard deviation、$`\beta>0`$ は
uncertainty の寄与を制御する parameter です。

## 3.3 Confidence bound と optimism

UCB は、posterior mean だけを見るのではなく、不確実な候補に対して楽観的な値を与えます。

$$
\sqrt{\beta}\sigma_n(x)
$$

が大きいほど、まだ十分に観測されていない点が候補になりやすくなります。

- 小さい $`\beta`$: exploitation 寄り。
- 大きい $`\beta`$: exploration 寄り。

この明示性は UCB の大きな特徴です。

標準偏差が正の正規分布の下で、$0<\delta<1/2$ に対し
$P(f(x)\le\mu_n(x)+\Phi^{-1}(1-\delta)\sigma_n(x)\mid\mathcal D_n)=1-\delta$ です。
したがって $\sqrt\beta=\Phi^{-1}(1-\delta)$ と置くとUCBは点ごとの上側分位点になります。
ただし点ごとの被覆率だけでは、探索全体の同時保証は得られません。

## 3.4 GP-UCBの理論と実用UCBを分ける

理論解析で使われるGP-UCBでは、beta scheduleは単なる固定tuning parameterではなく、
iteration、confidence level、探索空間、kernel / function class等に依存して設定されます。

このscheduleはregret theoremの仮定と結び付いています。したがって実装でbetaを大きくしただけで、
特定のGP-UCB theoremの保証が自動的に得られるわけではありません。

実務コードで固定値 $`\beta`$ を使う UCB と、regret bound を導く GP-UCB の理論的 schedule は区別する必要があります。

したがって、

```text
UCB acquisition の beta
```

と

```text
特定の GP-UCB theorem が要求する beta_t
```

を同一視すべきではありません。

例えば候補集合が固定した有限集合で点数が $M$、条件付き正規事後分布が正しく、
ハイパーパラメータを既知とするベイズ設定を考えます。
正規尾確率の上界 $P(|Z|>a)\le2e^{-a^2/2}$ と和集合上界を使えば、

$$
\beta_t=2\log\left(\frac{M\pi^2t^2}{3\delta}\right)
$$

により、全候補・全時刻についての区間の失敗確率を合計で $\delta$ 以下にできます。
時刻 $t$ の失敗確率を $6\delta/(\pi^2t^2)$ 以下に配分し、その和が $\delta$ になるためです。
これは有限集合と整合的な事後分布の下での一例であり、連続領域や再推定した任意のカーネルにそのまま適用する保証ではありません。

この同時区間が成立し、各時刻のUCBを厳密に最大化できれば、瞬時リグレットは
$f(x^*)-f(x_t)\le2\sqrt{\beta_t}\,\sigma_{t-1}(x_t)$ と評価できます。
最適点を上から覆う区間と、選択点を下から覆う区間を比較することで得られます。
累積リグレットを抑えるには、さらに選択点での分散の総和を評価する必要があります。

## 3.5 Lower Confidence Bound

最小化問題では

$$
\operatorname{LCB}(x)
=
\mu_n(x)-\sqrt{\beta}\,\sigma_n(x)
$$

を最小化する形で考えられます。

一方、実装では objective の符号を反転して最大化問題として統一することもあります。重要なのは optimization convention を一貫させることです。

## 3.6 EI との違い

EI は improvement の期待値を計算します。UCB は posterior の location と scale を直接組み合わせます。

```text
EI
    現在の基準値をどれだけ改善するか

UCB
    posterior mean と uncertainty の楽観的上界
```

UCB は $`f_{\mathrm{best}}`$ を直接必要とせず、exploration strength を parameter で明示的に調整できる点が特徴です。

## 3.7 qUCBはUCB上位q点ではない

batch settingでは、複数候補のjoint posterior samplesからbatch utilityを構成するqUCB系を
利用できます。

qUCBをpointwise UCBの上位q点として理解するのは適切ではありません。
batch acquisitionではcandidate間のdependenceを保持したposterior samplingとjoint utilityが
候補集合の価値へ影響します。

一つのバッチへの拡張は、同時正規標本 $F_j$ に対して

$$
\alpha_{qUCB}(X)=\mathbb E\left[
\max_{j\le q}\left\{\mu_j+\sqrt{\frac{\beta\pi}{2}}\,|F_j-\mu_j|\right\}\right]
$$

と定義するものです。$q=1$ では $\mathbb E|F-\mu|=\sigma\sqrt{2/\pi}$ なので単点UCBへ戻ります。
候補間の依存は最大値の期待値に作用するため、周辺のUCB値の順位だけでは評価できません。
この拡張が単点の信頼保証をそのまま集合の保証へ変えるわけでもありません。

q-acquisition の一般的な意味は [Batch / Noisy](04_batch_noisy.md) を参照してください。

## 3.8 Posterior calibration の影響

UCB は uncertainty を式へ直接入れるため、posterior scale の calibration に特に敏感です。

- $`\sigma(x)`$ の過小評価 → exploration が弱くなる。
- $`\sigma(x)`$ の過大評価 → exploration が過剰になる。

これは EI など他の acquisition にも影響しますが、UCB では $`\sqrt\beta\,\sigma(x)`$ の形で特に見えやすくなります。

## 3.11 この章で覚えておくこと

UCBは事後平均へ標準偏差に比例した楽観的な補正を加えます。
係数の平方根の規約を確認し、点ごとの正規分位点と、探索全体で成立する信頼区間を分けます。
固定係数の数値的な順位付けだけで、特定のリグレット保証が得られるわけではありません。

最小化ではLCBや符号反転を一貫して使います。バッチへの拡張は候補の同時分布を必要とし、
単点のUCBの上位を取る方法や、単点の信頼保証そのものと同一ではありません。

## 3.12 まとめ

Confidence-bound family の中心的な考え方は、posterior uncertainty を明示的な optimism として意思決定へ変換することです。

理論的な GP-UCB の confidence schedule と、実務上 tuning する UCB parameter を区別すること、posterior uncertainty の
calibration を確認することが重要です。

## References

1. Srinivas, N. et al. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and
   Experimental Design. *Proceedings of ICML*.
2. Balandat, M. et al. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
   *Advances in Neural Information Processing Systems 33*.
3. Wilson, J., Hutter, F., and Deisenroth, M. (2018). Maximizing Acquisition Functions for Bayesian
   Optimization. *Advances in Neural Information Processing Systems 31*.

# 3. Confidence-bound Acquisition

## 3.1 UCBを一言で理解する

~~~text
UCB = 現在よさそうな候補 + まだ分からない候補へのoptimism bonus
~~~

EIがimprovementをutilityにするのに対し、UCBはposterior locationとuncertaintyを
直接組み合わせて候補を順位付けします。

## 3.3 平均と不確実性を直接組み合わせる

Confidence-bound acquisition は、posterior mean と posterior uncertainty を明示的に組み合わせます。

最大化問題の代表的な Upper Confidence Bound（UCB）は

$
\operatorname{UCB}(x)
=
\mu_n(x)+\sqrt{\beta}\,\sigma_n(x)
$

です。

ここで \(\mu_n(x)\) は posterior mean、\(\sigma_n(x)\) は posterior standard deviation、$`\beta>0`$ は
uncertainty の寄与を制御する parameter です。

## 3.2 Confidence bound と optimism

UCB は、posterior mean だけを見るのではなく、不確実な候補に対して楽観的な値を与えます。

$
\sqrt{\beta}\sigma_n(x)
$

が大きいほど、まだ十分に観測されていない点が候補になりやすくなります。

- 小さい $`\beta`$: exploitation 寄り。
- 大きい $`\beta`$: exploration 寄り。

この明示性は UCB の大きな特徴です。

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

## 3.5 Lower Confidence Bound

最小化問題では

$
\operatorname{LCB}(x)
=
\mu_n(x)-\sqrt{\beta}\,\sigma_n(x)
$

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

このためanalytic q=1 UCBがmarginal mean / varianceで評価できることと、MC qUCBが要求する
posterior sampling capabilityを同一視しません。

q-acquisition の一般的な意味は [Batch / Noisy](04_batch_noisy.md) を参照してください。

## 3.8 Posterior calibration の影響

UCB は uncertainty を式へ直接入れるため、posterior scale の calibration に特に敏感です。

- \(\sigma(x)\) の過小評価 → exploration が弱くなる。
- \(\sigma(x)\) の過大評価 → exploration が過剰になる。

これは EI など他の acquisition にも影響しますが、UCB では \(\beta\sigma(x)\) の形で特に見えやすくなります。

## 3.9 beta の意味を API と混同しない

文献によって

$
\mu(x)+\sqrt{\beta}\sigma(x)
$

と書く場合と、

$
\mu(x)+\beta\sigma(x)
$

と書く場合があります。

したがって parameter 名だけで探索強度を比較せず、**係数が式のどこへ入る定義なのか**を確認する必要があります。BoTorch API を使う場合も、そのバージョンの定義を確認します。

## 3.10 BoTorch / robotorchan との対応

BoTorch は analytic UCB と MC batch variant を提供しています。robotorchan は標準 UCB をローカルに再実装せず、BoTorch
native path を利用します。

利用方法は [Standard acquisition](../../optimization/standard_acquisition.md)、現在の統合範囲は [Acquisition
integration status](../../optimization/acquisition-integration.md) を参照してください。

### 3.9.1 現在の qUCB registry contract

現在のregistryでは `qUpperConfidenceBound` をBoTorch-native Monte Carlo acquisitionとして
扱います。robotorchan独自のqUCB数式やwrapperを追加しているわけではありません。

integration guideではanalytic UCBもnative pathとして利用対象です。registryはBoTorchの全acquisition
catalogではないため、registry entryの有無とnative APIの存在を区別します。

したがって model compatibility は「GPかどうか」だけでなく、
posterior sampling capability を満たすかで判定します。

## 3.11 この章で覚えておくこと

- UCBはposterior meanにuncertainty由来のoptimism bonusを加える考え方
- betaはexploration strengthに関係するが、定義のparameterizationを確認する
- 固定betaの実用UCBとtheorem用GP-UCB scheduleを同一視しない
- LCBと符号反転によるmaximize conventionを混同しない
- qUCBはpointwise UCB上位q点ではなくjoint batch acquisition
- q=1 analytic UCBとMC qUCBではposterior requirementが異なり得る
- uncertainty scaleの意味とcalibrationはmodel familyに依存する
- robotorchanは標準UCB/qUCBを再実装せずBoTorch-native pathを使う

## 3.12 まとめ

Confidence-bound family の中心的な考え方は、posterior uncertainty を明示的な optimism として意思決定へ変換することです。

理論的な GP-UCB の confidence schedule と、実務上 tuning する UCB parameter を区別すること、posterior uncertainty の
calibration を確認することが重要です。

## References

1. Srinivas, N. et al. (2010). Gaussian Process Optimization in the Bandit Setting: No Regret and
   Experimental Design. *Proceedings of ICML*.
2. Balandat, M. et al. (2020). BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.
   *Advances in Neural Information Processing Systems 33*.

# 3. Confidence-bound Acquisition

## 3.1 平均と不確実性を直接組み合わせる

Confidence-bound acquisition は、posterior mean と posterior uncertainty を明示的に組み合わせます。

最大化問題の代表的な Upper Confidence Bound（UCB）は

\[
\operatorname{UCB}(x)
=
\mu_n(x)+\sqrt{\beta}\,\sigma_n(x)
\]

です。

ここで \(\mu_n(x)\) は posterior mean、\(\sigma_n(x)\) は posterior standard deviation、\(\beta>0\) は uncertainty の寄与を制御する parameter です。

## 3.2 Confidence bound と optimism

UCB は、posterior mean だけを見るのではなく、不確実な候補に対して楽観的な値を与えます。

\[
\sqrt{\beta}\sigma_n(x)
\]

が大きいほど、まだ十分に観測されていない点が候補になりやすくなります。

- 小さい \(\beta\): exploitation 寄り。
- 大きい \(\beta\): exploration 寄り。

この明示性は UCB の大きな特徴です。

## 3.3 GP-UCB の理論的な beta

理論解析で使われる GP-UCB では、\(\beta_t\) は単なる固定 tuning parameter ではなく、iteration \(t\)、信頼水準、探索空間、kernel / function class などに依存する confidence sequence として設定される場合があります。

実務コードで固定値 \(\beta\) を使う UCB と、regret bound を導く GP-UCB の理論的 schedule は区別する必要があります。

したがって、

```text
UCB acquisition の beta
```

と

```text
特定の GP-UCB theorem が要求する beta_t
```

を同一視すべきではありません。

## 3.4 Lower Confidence Bound

最小化問題では

\[
\operatorname{LCB}(x)
=
\mu_n(x)-\sqrt{\beta}\,\sigma_n(x)
\]

を最小化する形で考えられます。

一方、実装では objective の符号を反転して最大化問題として統一することもあります。重要なのは optimization convention を一貫させることです。

## 3.5 EI との違い

EI は improvement の期待値を計算します。UCB は posterior の location と scale を直接組み合わせます。

```text
EI
    現在の基準値をどれだけ改善するか

UCB
    posterior mean と uncertainty の楽観的上界
```

UCB は \(f_{\mathrm{best}}\) を直接必要とせず、exploration strength を parameter で明示的に調整できる点が特徴です。

## 3.6 qUCB

batch setting では、複数候補の joint posterior を扱う qUCB 系を利用できます。

qUCB を pointwise UCB の上位 q 点として理解するのは適切ではありません。batch acquisition では候補間 correlation と joint sample utility が候補集合の価値へ影響します。

q-acquisition の一般的な意味は [Batch / Noisy](04_batch_noisy.md) を参照してください。

## 3.7 Posterior calibration の影響

UCB は uncertainty を式へ直接入れるため、posterior scale の calibration に特に敏感です。

- \(\sigma(x)\) の過小評価 → exploration が弱くなる。
- \(\sigma(x)\) の過大評価 → exploration が過剰になる。

これは EI など他の acquisition にも影響しますが、UCB では \(\beta\sigma(x)\) の形で特に見えやすくなります。

## 3.8 beta の意味を API と混同しない

文献によって

\[
\mu(x)+\sqrt{\beta}\sigma(x)
\]

と書く場合と、

\[
\mu(x)+\beta\sigma(x)
\]

と書く場合があります。

したがって parameter 名だけで探索強度を比較せず、**係数が式のどこへ入る定義なのか**を確認する必要があります。BoTorch API を使う場合も、そのバージョンの定義を確認します。

## 3.9 BoTorch / robotorchan との対応

BoTorch は analytic UCB と MC batch variant を提供しています。robotorchan は標準 UCB をローカルに再実装せず、BoTorch native path を利用します。

利用方法は [Standard acquisition](../../optimization/standard_acquisition.md)、現在の統合範囲は [Acquisition integration status](../../optimization/acquisition-status.md) を参照してください。

## 3.10 まとめ

Confidence-bound family の中心的な考え方は、posterior uncertainty を明示的な optimism として意思決定へ変換することです。

理論的な GP-UCB の confidence schedule と、実務上 tuning する UCB parameter を区別すること、posterior uncertainty の calibration を確認することが重要です。

# Transfer Bayesian Optimization（転移ベイズ最適化）

## 1. 問題設定

ターゲットタスクの未知目的関数 \(f_t(x)\) を少数の評価で最適化するため、
関連するソースタスク \(f_1,\ldots,f_K\) の既存観測や学習済みモデルを利用する。
ソースとターゲットの最適点や応答面が異なる場合、**負の転移**が生じ得る。

## 2. Warm-start

過去の良好な入力をターゲットの初期評価候補として再利用する。
ソースの最良点がターゲットでも良好である保証はない。
候補選択による転移と、事後分布や獲得関数による転移は区別する。

## 3. Ranking-weighted Gaussian Process Ensemble（RGPE）

各ソースモデル \(M_i\) とターゲットモデル \(M_t\) のランキング損失を比較し、
ブートストラップでモデル重み \(w_i\) を推定する。
\(w_i\ge 0\)、\(\sum_i w_i=1\) とし、ターゲットを含むモデル集合を統合する。

モデル平均の一例は

\[
\bar\mu(x)=\sum_i w_i\mu_i(x)
\]

であり、モーメント整合による分散は

\[
\bar\sigma^2(x)=\sum_i w_i[\sigma_i^2(x)+\mu_i^2(x)]-\bar\mu^2(x).
\]

ソースモデルの順位予測がターゲットと合わない場合、その重みを小さくすることで
負の転移を抑制することを目指す。ただし完全な回避は保証されない。

## 4. Transfer Acquisition Function（TAF）

ターゲットの獲得関数とソースの情報を、候補点の評価段階で組み合わせる。
実装可能な一形式は、最大化問題で

\[
a_{\mathrm{TAF}}(x)=w_t\operatorname{EI}_t(x)
+\sum_{i=1}^{K}w_i\max(\mu_i(x)-f_t^*,0)
\]

となる。\(f_t^*\) はターゲットで観測した最良値である。
最小化では改善量の符号を反転する。

これはソース予測**平均**による改善量を使う形式であり、
ソース予測分布全体に対するEIやRGPEの事後分布統合と同一ではない。
固定重みTAFは、誤ったソースに重みを残すと負の転移が起こり得る。

## 5. 評価

ターゲットの評価回数を横軸に、最良観測値（best-so-far）を比較する。
正の転移と負の転移の両方を評価し、複数乱数シードでばらつきを確認する。
計算コストの比較では、RGPEのLOO再学習などの追加処理も考慮する。

## 参考文献

- Feurer, Letham, Bakshy (2018), *Scalable Meta-Learning for Bayesian Optimization using Ranking-Weighted Gaussian Process Ensembles*.
- Wistuba, Schilling, Schmidt-Thieme (2018), *Scalable Gaussian Process-Based Transfer Surrogates for Hyperparameter Optimization*.

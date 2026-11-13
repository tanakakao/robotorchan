# Transfer Bayesian Optimization（転移ベイズ最適化）

## 1. 問題設定と転移の考え方

ターゲットタスクの未知目的関数 \(f_t:\mathcal X\to\mathbb R\) を、
限られた評価回数で最大化する。既知のソースタスク
\(f_1,\ldots,f_K\) について、過去の観測データまたは学習済みの
確率モデルが利用できるとする。ターゲットの観測データは

\[
D_t=\{(x_j,y_j)\}_{j=1}^{n_t},\qquad
y_j=f_t(x_j)+\epsilon_j
\]

である。通常のベイズ最適化は \(D_t\) のみで代理モデルを更新するが、
転移ベイズ最適化はソースの情報を使い、初期探索や獲得関数の効率改善を目指す。

転移方法は大きく三つに分けられる。

| 手法 | ソース情報の利用箇所 | ターゲット側で更新するもの |
| --- | --- | --- |
| Warm-start | 初期評価点の選択 | 初期観測データ |
| RGPE | 複数モデルの事後予測の統合 | ターゲットGP・ランキング重み |
| TAF | 獲得関数へのソース予測の加算 | ターゲットGP（ソース重みは固定） |

ソースとターゲットの応答面が異なると、誤った領域を優先する
**負の転移（negative transfer）** が起こる。ソースの最適点が
ターゲットでも最適とは限らず、転移手法は常に通常BOを上回るわけではない。

以下では最大化を基本とし、最小化では改善量の向きを反転する。

## 2. 基本となるGaussian ProcessとExpected Improvement

GPの事後予測を

\[
f_t(x)\mid D_t\sim\mathcal N(\mu_t(x),\sigma_t^2(x))
\]

とする。既知の最良ターゲット観測値を \(f_t^*=\max_j y_j\) とすると、
1点での改善量とExpected Improvement（EI）は

\[
I(x)=\max(f_t(x)-f_t^*,0),\qquad
\operatorname{EI}_t(x)=\mathbb E[I(x)\mid D_t].
\]

\(\sigma_t(x)>0\) なら

\[
\operatorname{EI}_t(x)=
(\mu_t(x)-f_t^*)\Phi(z)+\sigma_t(x)\phi(z),
\qquad z=\frac{\mu_t(x)-f_t^*}{\sigma_t(x)}
\]

となる。\(\Phi\) と \(\phi\) は標準正規分布の累積分布関数と密度関数である。
EIは予測平均が高い点と、不確実性が大きい点の両方を評価する。
本ライブラリの候補提案ではBoTorchのEIを再利用する。

## 3. Warm-start：過去の良好な入力を初期点にする

ソース観測点の目的値、またはソースGPの予測平均に基づいて
候補を順位付けし、ターゲットの初期評価点を選ぶ。
必要な点数が足りなければSobol系列から補完する。

重要なのは、ソースの目的値 \(f_i(x)\) を
**ターゲットの観測値 \(f_t(x)\) として転用しない**ことである。
選ばれた \(x\) は必ずターゲット目的関数で評価する。
Warm-startは初期点の選択法であり、RGPEのモデル統合や
TAFの獲得関数統合とは異なる。

Mixed空間では、許可されたカテゴリ割当を
`fixed_features_list` で指定して候補を選ぶ。

## 4. RGPE：ランキング損失によるモデル重み

### 4.1 なぜ予測値の誤差ではなく順位を使うか

ソースとターゲットの目的値のスケールが異なっても、
「どちらの候補が良いか」という順序が一致する場合がある。
RGPEはモデルの予測がターゲット観測の順位を再現するかを評価する。

モデル \(M_i\) のターゲット観測点における予測平均を
\(\hat y_{i,j}\) とする。観測値が異なるペアの集合を
\(P=\{(j,k):j<k,\ y_j\ne y_k\}\) とし、ランキング損失を

\[
L_i=\frac{1}{|P|}\sum_{(j,k)\in P}
\mathbf 1\left[
\operatorname{sign}(\hat y_{i,j}-\hat y_{i,k})
\ne\operatorname{sign}(y_j-y_k)
\right]
\]

と定義する。予測値が同点でも観測順位と一致しなければ誤りと数える。

### 4.2 ターゲットモデルのLeave-One-Out（LOO）

ターゲットGPが学習に使った点でそのまま予測すると、
学習データに適合し過ぎて公平な比較にならない。
そこでターゲットモデルについては各点を除いて再学習し、
除いた点の予測平均を集める。

\[
\hat y_{t,j}=
\mathbb E[f_t(x_j)\mid D_t\setminus\{(x_j,y_j)\}].
\]

ソースモデルにはターゲット観測点でのposterior meanを使う。
このためRGPEは通常の単一GPによるBOより学習コストが高い。

### 4.3 ブートストラップと負の転移の抑制

本実装では、まず全ペアに対する損失がターゲットモデルの損失を
超えるソースモデルを重み推定から除外する。
残るモデルについて観測ペアを復元抽出し、各反復で
ランキング損失が最小のモデルに勝者票を与える。
同点の勝者には票を等分する。

反復回数を \(B\)、反復 \(b\) におけるモデル \(i\) の票を
\(v_i^{(b)}\) とすると、

\[
w_i=\frac{1}{B}\sum_{b=1}^{B}v_i^{(b)},
\qquad w_i\ge0,\quad\sum_i w_i=1.
\]

重みの順序は **ソースモデル群、最後にターゲットモデル** である。
この除外規則は負の転移の抑制策だが、完全な防止保証ではない。
観測値がすべて同点で順位情報がない場合はターゲットに全重みを与える。

### 4.4 事後分布のモーメント整合

モデル \(M_i\) の候補集合 \(X\) における予測平均ベクトルを
\(\boldsymbol\mu_i(X)\)、共分散行列を \(\Sigma_i(X)\) とする。
重み付き混合分布の平均と共分散は

\[
\bar{\boldsymbol\mu}=\sum_iw_i\boldsymbol\mu_i,\qquad
\bar\Sigma=\sum_iw_i\Sigma_i+
\sum_iw_i(\boldsymbol\mu_i-\bar{\boldsymbol\mu})
(\boldsymbol\mu_i-\bar{\boldsymbol\mu})^\top.
\]

第2項はモデル間の予測平均の不一致による不確実性である。
アンサンブル事後分布はこの二つのモーメントに一致する**単一Gaussian分布**を返す。
これは厳密なGaussian mixtureそのものではなく近似である。

1点提案にはこの近似posteriorに対する解析的EIを使い、
Jointバッチでは同じ近似posteriorからのMC qEIを使う。

## 5. TAF：獲得関数でソース情報を利用する

TAFはRGPEと違い、GPのposterior自体を統合しない。
ターゲットEIに、各ソースモデルの予測平均による改善量を足す。

\[
a_{\mathrm{TAF}}(x)=
w_t\operatorname{EI}_t(x)+
\sum_{i=1}^{K}w_i\max(\mu_i(x)-f_t^*,0).
\]

ここで \(w_i\ge0\)、\(\sum_iw_i+w_t=1\) とし、
**重みは利用者が明示指定**する。ソースの改善量は
\(\mathbb E[\max(f_i(x)-f_t^*,0)]\) ではなく、
\(\max(\mathbb E[f_i(x)]-f_t^*,0)\) である。
したがってソースの予測不確実性は直接は反映されない。

ソース重みを大きくすると過去タスクの良好な領域を優先しやすいが、
ターゲットとの関連が弱い場合は負の転移を生む。
RGPEのランキング重みをTAFが自動推定するわけではない。

## 6. Mixed入力の扱い

探索空間を連続部分 \(\mathcal X_c\) とカテゴリ部分
\(\mathcal X_d\) の直積として

\[
\mathcal X=\mathcal X_c\times\mathcal X_d
\]

と表す。カテゴリを整数コードで入力しても、コード間の差を
連続距離として解釈すべきではない。
本実装ではカテゴリ列を `cat_dims` で指定して
BoTorchのMixed入力対応GPを用い、
`fixed_features_list` に許可カテゴリの割当を列挙して
`optimize_acqf_mixed` に候補探索を委譲する。

ソース・ターゲットモデルは同じ入力表現と次元を使う必要がある。
候補間制約や任意の組合せ制約まで自動的に扱うという意味ではない。

## 7. q>1：Sequential greedyとJoint batch

### 7.1 Sequential greedy

1点ずつ解析的な獲得関数を最適化する。
次の点を選ぶ前に、直前の候補 \(x_k\) のターゲットGP事後平均

\[
\tilde y_k=\mu_t(x_k)
\]

を仮想観測として条件付ける。実測値を得る前に候補を分散させる
近似的な方法である。RGPEのランキング重み、TAFの指定重みは
バッチ内では更新しない。実装では仮想値をbest-so-farにも反映する。

### 7.2 Joint batchのMC qEI

\(X=(x_1,\ldots,x_q)\) を一括で選ぶ。ターゲットのqEIは

\[
\operatorname{qEI}_t(X)=
\mathbb E\left[
\max\left(\max_{1\le j\le q}f_t(x_j)-f_t^*,0\right)
\mid D_t\right].
\]

候補間の同時事後共分散を考慮するため、
各点のEIの単純和とは異なる。実装はBoTorchの
`qExpectedImprovement` とSobol QMCサンプラーを利用する。

RGPEではモーメント整合したensemble posteriorをqEIに渡す。
TAFのJoint版は

\[
a_{\mathrm{qTAF}}(X)=
w_t\operatorname{qEI}_t(X)+
\sum_{i=1}^{K}w_i
\max\left(\max_{1\le j\le q}\mu_i(x_j)-f_t^*,0\right).
\]

ソース項は**最大の予測平均改善量**であり、
ソースposteriorからのMCサンプリングは行わない。
Jointはバッチ全体を最適化するため計算量が増えやすいが、
Sequentialの仮想観測近似を使わず候補間相関を直接扱う。

最小化ではターゲット目的値を符号反転してqEIを計算し、
ソース項も \(f_t^*-\mu_i(x_j)\) を改善量として使う。

## 8. 実験設計と適用上の注意

通常BO、Warm-start、RGPE、TAFを、同じ初期観測数・
ターゲット評価予算・乱数シードで比較する。
横軸はターゲットの実評価回数、縦軸はbest-so-farとする。
複数シードの分布と、モデル学習・候補最適化の計算時間も報告する。

- **正の転移**：ソースとターゲットの良好な領域が近い場合。
- **負の転移**：最適領域や順位が異なるソースを使う場合。
- **Mixed**：許可カテゴリ割当の範囲とカテゴリ別の探索偏りを確認する。
- **q>1**：SequentialとJointの改善量、候補の重複、実行時間を比較する。

対象はスカラー出力の単目的最適化である。モデル間の入力表現の一致、
ソースの関連性、予測の較正は利用者が確認する必要がある。
候補提案関数は目的関数を評価しないため、実際の観測値を得た後で
ターゲットデータを更新する。

## 参考文献

- Feurer, Letham, Bakshy (2018), *Scalable Meta-Learning for Bayesian Optimization using Ranking-Weighted Gaussian Process Ensembles*.
- Wistuba, Schilling, Schmidt-Thieme (2018), *Scalable Gaussian Process-Based Transfer Surrogates for Hyperparameter Optimization*.
- Jones, Schonlau, Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions*.
- Wilson, Hutter, Deisenroth (2018), *Maximizing Acquisition Functions for Bayesian Optimization*.

実装対応とNotebookへのリンクは
[Transfer BO利用ガイド](../guide/26_transfer_bayesian_optimization.md)を参照する。

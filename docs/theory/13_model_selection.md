# 13. Model Selection

この章では、これまで説明してきた Gaussian Process（GP）モデルを、**問題設定から逆算して選ぶ方法**を整理します。

しかし、モデル数が増えるほど重要になるのは「最も高度なモデルを使うこと」ではなく、

> **データ生成過程に必要な構造だけをモデルへ入れること**

です。

複雑なモデルには、より強い仮定、より多いハイパーパラメータ、より高い計算コストが伴います。

したがって本章では、通常のscalar regressionなら 単一タスク GP を比較基準に置きつつ、
**観測形式や構造が最初から異なる問題を無理に単一タスク GP へ変換しない**方針を採用します。

---

[GP章](02_gaussian_process.md)を基礎に、前章までの観測形式・共分散構造・推論方法を比較します。
本章のフローは候補を絞るための仮説整理であり、モデル名だけから性能の順序を決めるものではありません。

## 13.1 最初に覚える原則

モデル選択の基本原則は次です。

```text
まず 単一タスク GP を考える
        ↓
標準 GP では表現できない構造がある？
        ↓
      Yes
        ↓
その構造を直接表現するモデルへ移る
```

典型的な「標準 GP では足りない理由」は次です。

- カテゴリ変数がある
- fidelity がある
- task が複数ある
- output が複数ある
- task ごとに入力特徴量が異なる
- 観測値ではなく比較データである
- output が画像・スペクトル・時系列などの構造化出力である
- 高次元である
- データ件数が大きい
- 外れ値が問題になる
- 親変数によって有効な変数が変わる
- context decomposition がある

この「なぜ標準 GP では足りないのか」を言語化できない場合、まず 単一タスク GP をベースラインにするのが安全です。

---

### モデル名より先に7つの質問をする

初心者はモデル一覧から選ぶより、次の順で問題を分解すると判断しやすくなります。

~~~text
1. 何を観測する？        numeric / comparison
2. outputは何？          scalar / vector / tensor / function
3. inputに構造はある？   mixed / hierarchy / context
4. taskやfidelityはある？
5. noise生成過程は？     known / heteroskedastic / outlier / input uncertainty
6. d と n はどちらが難しい？
7. BOで必要なposterior操作は？
~~~

最後の7番は重要です。予測モデルとして使えることと、posterior sampling・fantasization・
特定のacquisitionでそのまま使えることは同義ではありません。

## 13.2 最重要フローチャート

```text
観測データは何か？
│
├─ pairwise comparison
│    └─ pairwise GP
│
└─ numeric response
     │
     ├─ structured output？
     │    ├─ tensor / grid → 高階 GP
     │    └─ explicit output coordinate T → 潜在座標の Kronecker GP
     │
     └─ scalar / ordinary output
          │
          ├─ conditional hierarchy？
          │    ├─ single task → Hierarchical Conditional カーネル GP
          │    └─ multi-task → Hierarchical Conditional カーネル マルチタスク GP
          │
          ├─ context decomposition？
          │    ├─ aggregated reward, independent context effect → context 加法 GP
          │    ├─ aggregated reward, learned context similarity → 加法的な context embedding GP
          │    └─ context-wise outputs observed → 乗法的な context embedding GP
          │
          ├─ multi-task / multi-output？
          │    ├─ independent outputs → 独立モデルの集合
          │    ├─ same X for all tasks → Kronecker 構造のマルチタスク GP
          │    ├─ different X, same feature structure → マルチタスク GP
          │    └─ task-specific feature structure → Heterogeneous MTGP
          │
          ├─ fidelity dimension？
          │    └─ 単一タスク Multi Fidelity GP
          │
          ├─ categorical variable？
          │    └─ 混合入力 GP
          │
          └─ continuous ordinary regression
               │
               ├─ high dimensional？
               │    ├─ sparse effective dimension → SAAS
               │    ├─ lighter SAAS approximation → MAP-SAAS
               │    └─ additive structure → 直交加法 GP
               │
               ├─ many observations？
               │    └─ 単一タスク Variational GP
               │
               ├─ strong sparse outliers？
               │    └─ Robust Relevance Pursuit 単一タスク GP
               │
               └─ otherwise → 単一タスク GP
```

この順序は絶対ではありません。

実際には、例えば「高次元かつ multi-task」「hierarchical かつ multi-task」のように複数の特徴が同時に存在します。

その場合は、**最も問題構造を規定している特徴から決める**のが重要です。

---

## 13.3 Step 1: 何を観測しているか

モデル選択で最初に確認するべきなのは、入力 `X` より先に **観測値の意味** です。

### 数値を直接観測している

例えば、

- 強度
- 歩留まり
- 電気抵抗
- 収率
- 寿命
- エネルギー

などです。

この場合、通常の regression GP が候補になります。

### 比較のみ観測している

例えば、

```text
A > B
B > C
A > C
```

のような preference / comparison data しか得られない場合、通常回帰へ無理に変換するべきではありません。

この場合は

```text
pairwise GP
```

がcomparison likelihoodを持つ候補です。通常のnumeric regressionとは観測modelが異なるため、
単なる0/1 regressionへの置換とは区別します。

詳しくは [Preference Learning](10_preference_learning.md) を参照してください。

---

## 13.4 Step 2: output は scalar か structured か

次に、1つの入力 `x` に対して何が返るかを確認します。

### scalar response

```text
x → y
```

これは通常の GP です。

### vector / tensor / function-valued response

```text
x → spectrum
x → time series
x → image
x → spatial field
```

この場合、各要素を独立 output として扱うよりも、output 内の相関構造を使う方が自然なことがあります。

### 高階 GP

出力そのものが tensor として自然に表現される場合です。

```text
観測応答
[n, d1, d2, ...]
```

画像、空間分布、多次元グリッドなどに向きます。

### 潜在座標の Kronecker GP

時間、波長、位置など、出力軸に明示的な座標 `T` が存在する場合です。

```text
f(x, t)
```

として扱います。

新しい `T` で予測したい場合には特に自然です。

詳しくは [Structured Output](11_structured_output.md) を参照してください。

---

## 13.5 Step 3: 探索空間に条件付き構造があるか

通常の Mixed variable と Hierarchical variable は区別する必要があります。

### Mixed variable

例えば

```text
temperature: continuous
pressure: continuous
catalyst: categorical
```

です。

全変数は常に意味を持ちます。

この場合、単一タスクの基本構成として、連続変数とカテゴリの意味を分ける混合入力GPを検討します。

### Hierarchical / conditional variable

例えば

```text
process_type = A
    → temperature active
    → pressure inactive

process_type = B
    → temperature inactive
    → pressure active
```

のように、親変数の値によって子変数の存在意味そのものが変わります。

この場合は

```text
Hierarchical Conditional カーネル GP
```

を使います。

複数 task がある場合は

```text
Hierarchical Conditional カーネル マルチタスク GP
```

です。

Mixed と Hierarchical を混同しないことが重要です。

詳しくは [Hierarchical / Contextual GP](12_hierarchical_contextual_gp.md) を参照してください。

---

## 13.6 Step 4: fidelity は存在するか

同じ設計点について、精度やコストの異なる評価方法が存在する場合を考えます。

例えば、

```text
low fidelity simulation
high fidelity simulation
experiment
```

です。

fidelity が「同じ物理量の精度レベル」を意味するなら、

```text
単一タスク Multi Fidelity GP
```

が候補になります。

重要なのは、単にデータソースが複数あるだけでは Multi-Fidelity とは限らないことです。

### Multi-Fidelity

```text
同じ target
異なる accuracy / cost
```

### Multi-task

```text
関連するが異なる task
```

この区別は非常に重要です。

詳しくは [Multi-Fidelity](06_multi_fidelity.md) を参照してください。

---

## 13.7 Step 5: 複数 task / output をどう扱うか

複数の response があるからといって、必ず マルチタスク GP を使うわけではありません。

最初に確認するのは

> output 間で情報共有したいか

です。

### 情報共有しない

```text
独立モデルの集合
```

が自然です。

例えば多目的 BO で、目的関数ごとに独立 GP を置く場合です。

### 情報共有する

次にデータ構造を見ます。

#### すべての task を同じ X で観測

```text
観測入力: [n, d]
観測応答: [n, m]
```

なら

```text
Kronecker 構造のマルチタスク GP
```

が候補です。

#### task ごとに異なる X で観測

long-format 表現が自然なら

```text
マルチタスク GP
```

です。

#### task ごとに feature set 自体が異なる

例えば

```text
task A: temperature, pressure, flow
task B: temperature, flow
```

なら

```text
Heterogeneous MTGP
```

を検討します。

詳しくは [Multi-task / Multi-output](07_multitask_multioutput.md) と [Hierarchical / Contextual
GP](12_hierarchical_contextual_gp.md) を参照してください。

---

## 13.8 Multi-objective と Multi-output は別概念

この区別は特に BO では重要です。

### Multi-output model

複数の予測値を持つモデル構造の問題です。

### Multi-objective BO

複数の最適化目的を同時に扱う意思決定問題です。

例えば

```text
strength
cost
```

を同時最適化する場合、2つの目的を 独立モデルの集合 で独立にモデル化して qEHVI を使うこともできます。

したがって

```text
Multi-objective = マルチタスク GP
```

ではありません。

モデルと acquisition objective は分離して考えます。

---

## 13.9 Step 6: context decomposition があるか

問題が複数の context block に分解できることがあります。

例えば

```text
context A → variables x_A
context B → variables x_B
```

です。

このとき、観測の仕方によってモデルが変わります。

### aggregated reward のみ観測

```text
y = y_A + y_B
```

のような合計値しか観測できない場合です。

context 間の明示的な類似性を学習しないなら

```text
context 加法 GP
```

を検討します。

context 間の latent similarity を学習したいなら

```text
加法的な context embedding GP
```

です。

### context-wise reward を個別観測

```text
y_A
y_B
```

をそれぞれ観測できるなら

```text
乗法的な context embedding GP
```

が候補です。

---

## 13.10 Step 7: 入力次元は高いか

標準 GP は理論上、高次元入力でも使えます。

しかし BO では、

- surrogate fitting
- lengthscale estimation
- acquisition optimization

のすべてが難しくなります。

そのため「高次元だから何を仮定できるか」を考えます。

### 有効な次元が少ない

```text
SAAS fully Bayesian 単一タスク GP
```

が強力な候補です。

SAAS は多くの変数をほぼ無関係とみなし、少数の active dimensions を選択する方向に prior を置きます。

### 高次元 + multi-task

```text
SAAS fully Bayesian マルチタスク GP
```

を検討します。

### Fully Bayesian SAAS が重い

```text
Additive Map SAAS 単一タスク GP
Ensemble Map SAAS 単一タスク GP
```

が軽量な候補になります。

### 加法構造が妥当

```text
f(x) ≈ f1(x1) + f2(x2) + ...
```

のような構造が期待できるなら

```text
直交加法 GP
```

も候補です。

詳しくは [High-dimensional GP](08_high_dimensional_gp.md) を参照してください。

---

## 13.11 高次元なら必ず SAAS ではない

高次元問題では、まず高次元性の原因を確認します。

### 本当に多数の独立設計変数がある

SAAS が適する可能性があります。

### One-hot encoding によって見かけ上高次元になった

Mixed / categorical structure を直接扱う方が自然な可能性があります。

### スペクトルや画像を flatten した結果、高次元になった

Structured Output / feature extraction の問題かもしれません。

### 多数の task / context を列として展開した

Multi-task / contextual model の方が自然かもしれません。

したがって

> 次元数だけを見て SAAS を選ばない

ことが重要です。

---

## 13.12 PCA などの次元削減との関係

入力次元削減と SAAS は異なるアプローチです。

### PCA

```text
X → Z
```

へ事前射影してから GP を学習します。

利点は計算量低減です。

一方で PCA は `Y` を見ずに分散の大きい方向を残すため、目的関数に重要な低分散方向を失う可能性があります。

### SAAS

元の入力空間で lengthscale sparsity を通じて relevant dimensions を学習します。

そのため、入力の意味を保持したまま variable relevance を扱いやすい利点があります。

次元削減を使う場合でも、探索途中で射影を頻繁に変更すると探索空間そのものが変化します。

固定 reducer と適応的 reducer は別設計として考えるべきです。

---

## 13.13 Step 8: データ件数は多いか

Exact GP の主要な制約は、学習点数 `n` に対する計算コストです。

標準的な Exact GP では、dense covariance の分解に概ね

$$
O(n^3)
$$

の計算量が現れます。

データ数が増え、Exact GP がボトルネックになる場合には

```text
単一タスク Variational GP
```

を検討します。

Variational GP は inducing points を使い、完全な covariance matrix を直接扱うコストを削減します。

ただし

> Variational GP は「高次元対策」ではなく主に「大規模 n 対策」

です。

高次元 `d` と大規模 `n` は別問題として考えます。

詳しくは [Variational GP](09_variational_gp.md) を参照してください。

---

## 13.14 高次元 d と大規模 n を分ける

これは頻繁に混同されます。

```text
高次元
d が大きい
→ SAAS / additive structure など

大規模データ
n が大きい
→ Variational GP など
```

例えば

```text
n = 40
d = 100
```

なら Exact GP の `n^3` はそれほど問題でなくても、高次元性による kernel estimation と acquisition optimization が問題になります。

逆に

```text
n = 10000
d = 5
```

なら入力次元は低くても Exact GP fitting が重くなります。

---

## 13.15 Step 9: noise生成過程とrobustnessを確認する

「外れ値があるか」だけではなく、**なぜ観測が不確かになるのか**を分けます。

~~~text
既知・固定の観測noise
  → 既知の観測分散 / likelihood

入力位置によってnoise varianceが変わる
  → heteroskedastic model

replicateからnoiseを推定できる
  → replicate-noise model

少数のcontamination / outlier
  → robust / contaminated model

入力そのものが揺らぐ
  → uncertain-input model

関数のsmoothnessやlengthscaleが場所で変わる
  → nonstationary model
~~~

これらは同じ「予測が不安定」という症状でも、仮定しているdata-generating processが異なります。

### noise

同じ `x` を繰り返しても少し異なる値が出る現象です。

### outlier

通常の noise model では説明しづらい極端な観測です。

### regime

異なる生成過程に属する観測です。

これらはモデル化が異なります。

```text
noise
→ likelihood / 既知の観測分散

outlier
→ robust / contaminated model

input uncertainty
→ uncertain-input model

nonstationarity
→ nonstationary model

regime
→ task / categorical / hierarchical model
```

---

## 13.17 モデル選択で最初に見る tensor shape

以下の形状はデータの数学的な配置を表す例です。$n$ は観測点数、$d$ は設計変数の数、
$m$ は文脈に応じて出力数または比較数を表します。タスク識別子が一列にあっても、
その列へ通常の連続距離を仮定するわけではありません。

モデル選択は数学だけでなく、データ shape を見るとかなり整理できます。

### 単一タスク GP

```text
観測入力: [n, d]
観測応答: [n, 1]
```

### Kronecker 構造のマルチタスク GP

```text
観測入力: [n, d]
観測応答: [n, m]
```

### マルチタスク GP

```text
観測入力: [n, d + 1]
                 ↑ task feature
観測応答: [n, 1]
```

### 独立モデルの集合

```text
model_1(X1, Y1)
model_2(X2, Y2)
...
```

### Heterogeneous MTGP

```text
観測入力s = [X_task0, X_task1, ...]
```

で、各 `X_task` の列数が異なり得ます。

### 高階 GP

```text
観測入力: [n, d]
観測応答: [n, d1, d2, ...]
```

### 潜在座標の Kronecker GP

```text
設計入力 X: [n, d]
出力座標 T: [M]
応答 Y: [n, M]（欠測がある場合は観測位置も区別）
```

です。

### pairwise GP

```text
datapoints: [n, d]
comparisons: [m, 2]
```

shape を先に整理すると、候補モデルをかなり絞れます。

---

## 13.18 問題設定から考える例

### 通常のプロセス条件最適化

```text
温度
圧力
時間
流量
→ 品質
```

基本候補:

```text
単一タスク GP
```

### 製法カテゴリを含む

```text
温度
圧力
製法A/B/C
→ 品質
```

基本候補:

```text
混合入力 GP
```

### 製法ごとに有効条件が違う

```text
製法A → 温度
製法B → 圧力
```

基本候補:

```text
Hierarchical Conditional カーネル GP
```

### 簡易シミュレーションと高精度シミュレーション

```text
単一タスク Multi Fidelity GP
```

### 複数装置間で情報共有

入力定義が同じなら

```text
マルチタスク GP
```

### 装置ごとにセンサー項目が異なる

```text
Heterogeneous MTGP
```

### 官能評価

```text
pairwise GP
```

### スペクトル全体を予測

波長座標を明示したいなら

```text
潜在座標の Kronecker GP
```

### 画像 response

```text
高階 GP
```

### 多数のプロセス変数から少数だけが効く

```text
SAAS fully Bayesian 単一タスク GP
```

### 大量の履歴データ

```text
単一タスク Variational GP
```

---

## 13.19 ベースライン比較を必ず行う

特殊モデルを採用するときも、可能なら 単一タスク GP と比較することを推奨します。

例えば SAAS を使うなら

```text
単一タスク GP
vs
SAAS
```

Multi-task を使うなら

```text
independent 単一タスク GP / 独立モデルの集合
vs
マルチタスク GP
```

Contextual model なら

```text
ordinary GP
vs
context 加法 GP / 加法的な context embedding GP
```

を比較します。

複雑なモデルが必ず良いとは限りません。

---

## 13.20 モデル選択指標

候補モデルは、学習データへの fit だけでなく predictive performance で比較します。

代表的には

- negative log predictive density
- RMSE / MAE
- rank correlation
- calibration
- interval coverage
- posterior variance の妥当性

があります。

BO ではさらに

- best-so-far
- simple regret
- cumulative regret
- hypervolume
- sample efficiency

なども重要です。

BOのためのモデル選択では、これらの予測指標に加えて、限られた予算での意思決定の成果を評価します。

### 平均の誤差と予測分布の評価を分ける

検証観測 $y_i$ に対する予測平均を $\mu_i$、**観測ノイズを含む**予測分散を $s_i^2>0$ とします。
正規予測分布の平均負対数予測密度（NLPD）は、密度の対数を取ることで

$$
\operatorname{NLPD}
=\frac1{N_v}\sum_{i=1}^{N_v}
\left[\frac12\log(2\pi s_i^2)+\frac{(y_i-\mu_i)^2}{2s_i^2}\right]
$$

となります。$N_v$ は検証観測数です。第一項は広すぎる分布、第二項は
分散に比べて大きい予測誤差を罰します。分散を増やすだけで改善できる指標ではありません。
例えば $y=1$、$\mu=0$ のとき、分散1なら約1.419、分散0.01なら約48.616です。
二つの予測平均の誤差は同じでも、後者は誤った予測に強い確信を与えています。
RMSEはこの違いを表しません。

一方、区間被覆率だけなら過度に広い区間でも高くなるため、区間幅も確認します。
潜在関数値が既知の試験問題を評価する場合は、観測ノイズを含む予測分布とは対象が異なります。
複数出力の同時予測を評価したい場合も、周辺のNLPDの平均だけでは相関の妥当性までは確認できません。

---

## 13.21 Cross-validation だけで決めない

通常の機械学習では cross-validation が中心ですが、BO ではデータが i.i.d. ではありません。

探索ループでは acquisition function により観測点が選ばれるため、データ分布は時間とともに変わります。

したがって

```text
offline CV
+
sequential BO benchmark
```

の両方を見るのが望ましいです。
過去の探索が集中した領域での誤差は、未探索領域の誤差を代表するとは限りません。
時間変化が重要なら時系列順、未知タスクへの転移が目的ならタスク単位、
未知の設計点で構造化出力を予測するなら入力点単位で検証を分割します。
同一入力の反復測定を学習・検証へ無作為に分散させると、未観測入力への一般化を過大評価し得ます。

前処理とハイパーパラメータの選択は各学習部分で行い、検証側の値を使って再調整し続けないようにします。
BOの逐次比較では初期データ、評価予算、ノイズ条件をそろえ、複数回の探索のばらつきも確認します。
評価コストが異なる場合は、観測回数だけでなく累積費用や経過時間をそろえて比較します。

---

## 13.22 uncertainty calibration を確認する

BO では posterior mean だけでなく posterior uncertainty が acquisition に直接使われます。

そのため

```text
予測平均が合っている
```

だけでは不十分です。

例えば UCB なら

$$
\alpha(x) = \mu(x) + \beta^{1/2}\sigma(x)
$$

なので、`σ(x)` が過小評価されると探索不足になります。

逆に過大評価されると過度に探索します。

モデル選択では uncertainty quality も確認します。

---

## 13.23 複雑なモデルを選ぶデメリット

特殊モデルにはメリットだけでなくコストがあります。

### パラメータが増える

過学習や identifiability の問題が増えます。

### fitting が難しくなる

局所解、数値不安定性、収束時間増加が起きます。

### acquisition optimization も難しくなる

hierarchical / high-dimensional / mixed spaces では surrogate だけでなく acquisition optimizer も複雑になります。

### debugging が難しい

異常の原因が data / model / optimizer のどこにあるか分かりにくくなります。

したがって

> 必要な複雑さだけを追加する

のが重要です。

---

## 13.24 モデルと acquisition function を分離する

Surrogate model と acquisition function は別の選択問題です。

例えば

```text
単一タスク GP + LogEI
単一タスク GP + UCB
単一タスク GP + KG
```

は同じ model でも探索方針が異なります。

Multi-objective では

```text
独立モデルの集合 + qLogEHVI
```

のような組み合わせもあります。

したがって

```text
モデル選択
↓
posterior をどう作るか

acquisition 選択
↓
posterior を使って次点をどう選ぶか
```

を分離して考えます。

詳しくは [Acquisition Function](04_acquisition_function.md) を参照してください。

---

## 13.25 モデルと search-space optimizer も分離する

例えば Mixed variable では

```text
混合入力 GP
```

は surrogate model です。

一方でカテゴリ組合せをどう探索するかは

離散割当ごとに連続領域を最適化する方法や、有限候補集合を探索する方法など、獲得関数の最適化側の問題です。

Hierarchical model でも同様に、conditional kernel が正しくても無効な candidate を生成してよいわけではありません。

```text
surrogate model
candidate validity
acquisition optimization
```

を別レイヤとして設計します。

---

## 13.26 「モデルが表現できる」と「BOで使いやすい」は別

ある model が posterior を返せても、実務 BO ではさらに確認が必要です。

- acquisition がその posterior を扱えるか
- objective transform が必要か
- batch q に対応するか
- constraint をどう入れるか
- candidate optimizer が探索空間を扱えるか
- pending observations を扱えるか

---

## 13.27 学習方法から見た分類

モデルは fitting 方法でも分類できます。

### Exact MLL

Gaussian 観測モデルでは、対数周辺尤度を最大化して hyperparameter を学習できます。

### Variational ELBO

```text
単一タスク Variational GP
```

は `変分下界（ELBO）` を使います。

### NUTS

```text
SAAS fully Bayesian 単一タスク GP
SAAS fully Bayesian マルチタスク GP
```

は Fully Bayesian inference を使います。

### Laplace approximation

```text
pairwise GP
```

では preference likelihood に対して Laplace approximation が使われます。

モデルを差し替えると fitting pipeline まで変わる可能性がある点に注意してください。

---

### 対応可能な操作だけでは最適モデルは決まらない

必要な予測・標本化・条件付けが数学的に定義されるかを確認することは有用ですが、

~~~text
capability = その操作・構造をサポートする
~~~

であって、

~~~text
capability = その問題に最適なモデルである
~~~

ではありません。

最終判断では、data-generating assumptionとvalidationを優先します。

## 13.28 診断フロー

モデルを学習した後は次の順で確認します。

```text
1. tensor shape は正しいか
2. transform は正しいか
3. posterior mean は妥当か
4. posterior variance は妥当か
5. train point 近傍で挙動は自然か
6. task / context correlation は自然か
7. lengthscale は極端でないか
8. acquisition landscape は不自然でないか
9. candidate が探索空間制約を満たすか
```

特に複雑モデルでは、candidate を見る前に posterior を可視化することを推奨します。

---

## 13.29 よくある誤選択 1: categorical を連続値として扱う

例えば

```text
material = 0, 1, 2
```

として RBF kernel に入れると、

```text
distance(0,1) < distance(0,2)
```

という順序関係を暗黙に仮定します。

名義カテゴリなら通常これは不適切です。

```text
混合入力 GP
```

を検討します。

---

## 13.30 よくある誤選択 2: multi-objective だから マルチタスク GP

多目的 BO で目的間相関を必ず学習する必要はありません。

目的ごとに独立 surrogate を持つ

```text
独立モデルの集合
```

は非常に自然な選択です。

マルチタスク GP は「複数目的だから」ではなく

> task 間の covariance を利用したい

ときに選びます。

---

## 13.31 よくある誤選択 3: 高次元だから Variational GP

Variational GP が主に解決するのは `n` の大きさです。

入力次元 `d` の大きさを直接解決するモデルではありません。

```text
d large → SAAS / additive / dimension reduction
n large → Variational GP
```

と整理します。

---

## 13.32 よくある誤選択 4: データソースが違うから Multi-Fidelity

異なる装置、研究所、製品は必ずしも fidelity ではありません。

fidelity は通常、同じ target に対する近似精度や評価コストの軸です。

異なる生成過程なら multi-task の方が自然な場合があります。

---

## 13.33 よくある誤選択 5: 欠損 feature を 0 埋めして通常 GP

Task A には存在しない feature を 0 埋めして通常 マルチタスク GP へ入れると、0 という値に物理的意味があるかのように扱われる可能性があります。

Task ごとに feature structure が本質的に違う場合は

```text
Heterogeneous MTGP
```

を検討します。

---

## 13.34 よくある誤選択 6: inactive hierarchical variable を適当な値で埋める

条件付き探索空間で inactive variable を 0.5 などで埋め、通常 kernel で距離計算すると、その無意味な値が covariance に影響します。

このような場合に

```text
Hierarchical Conditional カーネル GP
```

が有効です。

---

## 13.35 よくある誤選択 7: structured output を全部独立 GP

例えば 100 波長のスペクトルについて100個の独立 GP を作ることは可能です。

しかし波長方向の相関を捨てることになります。

波長という明示的座標があるなら

```text
潜在座標の Kronecker GP
```

を検討できます。

---

## 13.36 問題構造から見るモデル候補表

| 問題構造 | 基本候補 | 主な理由 |
|---|---|---|
| 通常の連続単一出力 | 単一タスク GP | 基準モデル |
| 連続 + categorical | 混合入力 GP | カテゴリ距離を適切に扱う |
| fidelity | 単一タスク Multi Fidelity GP | cost / fidelity correlation |
| related tasks | マルチタスク GP | task covariance |
| same X for every task | Kronecker 構造のマルチタスク GP | block / Kronecker structure |
| independent outputs | 独立モデルの集合 | 不要な相関を仮定しない |
| task-specific features | Heterogeneous MTGP | heterogeneous feature spaces |
| many observations | 単一タスク Variational GP | inducing-point approximation |
| preference comparisons | pairwise GP | latent utility model |
| high-d sparse effects | SAAS fully Bayesian 単一タスク GP | sparse effective dimension |
| high-d multitask | SAAS fully Bayesian マルチタスク GP | SAAS + task sharing |
| lighter SAAS | MAP-SAAS models | NUTS を避ける |
| additive high-d | 直交加法 GP | additive structure |
| sparse outliers | Robust Relevance Pursuit 単一タスク GP | robust modeling |
| tensor output | 高階 GP | output tensor covariance |
| explicit output axis | 潜在座標の Kronecker GP | X × T product space |
| conditional variables | Hierarchical Conditional カーネル GP | active dimensions |
| conditional + multitask | hierarchical conditional multi-task GP | hierarchy + task covariance |
| contextual additive reward | context 加法 GP | context decomposition |
| latent context similarity | 加法的な context embedding GP | learned embeddings |
| context-wise outputs | 乗法的な context embedding GP | contextual multi-output |

---

## 13.37 計算コストの観点

非常に粗い分類ですが、実務上は次の感覚を持っておくと便利です。

```text
軽い
│
├─ 単一タスク GP
├─ 混合入力 GP
├─ 独立モデルの集合（小規模）
├─ マルチタスク GP（小規模）
│
├─ structured / hierarchical / contextual exact models
│
├─ Variational GP（大規模向けだが最適化設定が増える）
│
├─ MAP-SAAS
│
└─ Fully Bayesian SAAS / NUTS

重い
```

ただし実際のコストは

- n
- d
- number of tasks
- q
- acquisition type
- number of restarts
- MC samples

などに大きく依存します。

---

## 13.38 データ量が少ないときほど仮定が重要

BO では通常、データが少ない状態から始まります。

複雑なモデルは自由度が高い一方、少数データでは十分に識別できないことがあります。

そのため domain knowledge に基づいて

```text
本当に task correlation があるか
本当に additive か
本当に fidelity ordering があるか
本当に context similarity があるか
```

を考えることが重要です。

---

## 13.39 探索初期と後半でモデルを変える考え方

モデルは BO 全期間で固定する必要はありません。

例えば探索初期は

```text
単一タスク GP
```

で始め、データが増えてから

```text
SAAS
マルチタスク GP
VariationalGP
```

へ移行する設計も可能です。

ただしモデル変更時には posterior と acquisition の挙動が大きく変わるため、比較・ログ保存が重要です。

---

## 13.40 自動モデル選択について

例えばデータ schema から

```text
categorical columns detected
→ 混合入力 GP candidate

task feature exists
→ マルチタスク GP candidate

very high d
→ SAAS candidate

large n
→ VariationalGP candidate
```

と候補を出すことはできます。

ただし完全自動選択には注意が必要です。

例えば「fidelity」「task」「context」は tensor shape だけでは判定できず、意味論が必要です。

そのため自動化するなら、データ形式から候補を提案し、観測の意味と科学的仮定を確認してから、
予測分布と逐次意思決定の両面で比較する手順が現実的です。

---

## 13.42 最終チェックリスト

モデルを決定する前に、次を確認してください。

### 観測

- scalar value か comparison か
- single output か structured output か

### 入力

- continuous か
- categorical があるか
- integer があるか
- hierarchy があるか
- fidelity があるか

### task / output

- task は複数か
- task 間で情報共有すべきか
- 全 task が同じ X か
- feature set は task 間で同じか

### scale

- `d` は大きいか
- `n` は大きいか
- sparse effective dimension を仮定できるか
- additive structure を仮定できるか

### reliability

- noise level は既知か
- sparse outlier があるか
- regime difference ではないか

### BO

- objective は scalar か multi-objective か
- acquisition は posterior type に対応するか
- optimizer は mixed / hierarchical space を扱えるか

---

## 13.43 最小モデルから始める

最後に最も重要な判断基準をもう一度示します。

```text
単一タスク GP で十分なら 単一タスク GP を使う
```

特殊モデルは

```text
「使えるから使う」
```

のではなく

```text
「標準 GP では表現できない構造があるから使う」
```

べきです。

この原則は、モデルの解釈性、数値安定性、debuggability、BO 全体の再現性を改善します。

---

## 13.44 この章で覚えておくこと

```text
普通の連続回帰
→ 単一タスク GP

categorical
→ 混合入力 GP

fidelity
→ 単一タスク Multi Fidelity GP

related tasks
→ マルチタスク GP / Kronecker 構造のマルチタスク GP

independent outputs
→ 独立モデルの集合

task-specific feature spaces
→ Heterogeneous MTGP

large n
→ 単一タスク Variational GP

preference
→ pairwise GP

high d + sparse effective dimensions
→ SAAS / MAP-SAAS

additive high-dimensional structure
→ 直交加法 GP

sparse outliers
→ Robust Relevance Pursuit 単一タスク GP

tensor output
→ 高階 GP

explicit output coordinates
→ 潜在座標の Kronecker GP

conditional active variables
→ Hierarchical Conditional カーネル GP

conditional + multi-task
→ Hierarchical Conditional カーネル マルチタスク GP

context decomposition + aggregated reward
→ context 加法 GP / 加法的な context embedding GP

context-wise output
→ 乗法的な context embedding GP
```
---

### 最終的な考え方

モデル選択は「最も高機能なclassを探す作業」ではありません。

~~~text
観測の意味
  ↓
data-generating structure
  ↓
必要なmodel assumption
  ↓
posterior contract
  ↓
acquisition / optimizerとの互換性
  ↓
validation
~~~

という順に絞ります。

複数の構造が同時にある場合は、それらの仮定を同時に満たすモデルを考え、予測分布と意思決定基準の適合性を検証します。

本章のフローで候補を絞った後は、診断で不足している仮定だけを追加し、同じ条件の比較で改善を確認します。

## 参考文献

1. Rasmussen, C. E., and Williams, C. K. I. (2006).
   *Gaussian Processes for Machine Learning*. MIT Press.
2. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and de Freitas, N. (2016).
   Taking the human out of the loop: A review of Bayesian optimization.
   *Proceedings of the IEEE*, 104(1), 148-175.
3. Balandat, M. et al. (2020).
   BoTorch: A framework for efficient Monte-Carlo Bayesian optimization.
   *Advances in Neural Information Processing Systems 33*.
4. [Gaussian Process](02_gaussian_process.md)
5. [High-dimensional GP](08_high_dimensional_gp.md)
6. [Robust Gaussian Process](14_robust_gaussian_process.md)
7. [Non-GP Surrogates](23_non_gp_surrogates.md)

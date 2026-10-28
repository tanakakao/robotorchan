# 11. Structured Output GP

## 11.1 この章で扱うこと

通常の Gaussian Process（GP）回帰では、1つの入力 `x` に対して1つの scalar output `y` を予測する問題を考えます。

しかし実際の科学・工学データでは、1回の実験から

- スペクトル
- 時系列
- 温度分布
- 画像
- 2次元マップ
- 3次元場
- 波長 × 時間の応答

のような **構造を持つ出力** が得られることがあります。

このような問題を structured output regression と考えます。

高階 GP と潜在座標の Kronecker 構造を持つ GP を中心に説明します。

本章では、

- structured output とは何か
- vector output と tensor output の違い
- separable covariance
- Kronecker product
- 高階 GP
- latent output coordinates
- 潜在座標の Kronecker GP
- missing outputs
- pathwise conditioning
- iterative inference
- structured output と Bayesian Optimization

を整理します。

---

### Multi-outputとの違いを最初に見る

重要なのは「出力数が多いか」ではなく、**出力の添字や座標に意味があるか**です。


~~~text
通常のmulti-output
x → [強度, 導電率, コスト]
     各成分は異なる量

structured output
x → [y(400 nm), y(410 nm), y(420 nm), ...]
     隣接成分が波長軸上で関係する
~~~

後者では、output indexを単なるラベルとして扱うより、波長・時間・空間などの
構造をcovarianceへ組み込む価値があります。

## 11.2 Structured Output とは

通常の scalar regression は


$$
x \mapsto y
$$

です。

multi-output regression なら


$$
x \mapsto \mathbf y
$$

です。

Structured Output では、さらに出力の並び方・位置関係そのものに意味があります。

例えばスペクトルなら


$$
y(x, \lambda)
$$

画像なら


$$
y(x, u, v)
$$

時空間場なら


$$
y(x,t,r)
$$

のように、出力側にも座標や軸があります。


```text
入力 x
  ↓
実験・シミュレーション
  ↓
出力 tensor

例:
32波長スペクトル      → [32]
64×64画像             → [64, 64]
20時間×50位置         → [20, 50]
16×16×16 volume       → [16, 16, 16]
```

重要なのは、出力要素同士が独立ではないことです。

---

## 11.3 出力を単純にflattenすればよいのか

例えば `64 x 64` 画像なら、flattenして4096出力として扱うことは理論上可能です。

しかし、flattenして各要素間の一般的なdense covarianceを直接表現しようとすると


$$
4096 \times 4096
$$

の巨大な covariance を考える必要があります。

さらに、単にflattenした表現を作っただけでは


```text
隣接pixelは似ている
波長が近いほど相関しやすい
時間的に近い点は似ている
```

という構造を積極的には利用できません。

Structured Output GP はこの構造を covariance に組み込みます。

---

## 11.4 Kronecker 構造

Structured Output GP の重要な道具が Kronecker product です。

2つの covariance matrix


$$
K_X
$$


$$
K_T
$$

から


$$
K = K_X \otimes K_T
$$

を作ると、入力側と出力軸側の covariance を分離して表現できます。

例えば


```text
K_X
  → 実験条件どうしの似方

K_T
  → 波長位置どうしの似方

K_X ⊗ K_T
  → 条件 × 波長の全体 covariance
```

です。

---

## 11.5 separable covariance

最も基本的な仮定は


$$
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
$$

です。

これは **separable covariance** の代表的な形です。

意味としては、


```text
全体の相関
=
入力条件の相関
×
出力座標の相関
```

です。

この仮定は強い一方、product-grid上ではKronecker線形代数を利用できる可能性があり、
巨大なdense covarianceをそのまま扱うより効率化できます。

ただし「separable kernelを書けば常に同じ計算量になる」という意味ではありません。
実際の効率は観測pattern、likelihood、solver、posterior計算にも依存します。

---

## 11.6 HOGP と 潜在座標の Kronecker GP の違い


```text
高階 GP
  └─ 出力 tensor の shape / 各軸の構造を利用する

潜在座標の Kronecker GP
  └─ 入力 X と明示的な出力座標 T の積空間をモデル化する
```

より具体的には、HOGPはtensor shapeの各axisを構造として扱い、
LatentKroneckerGPは `X × T` のproduct spaceを明示します。

これは優劣ではなく**output structureの表現方法の違い**です。

---

# 高階 GP

## 11.7 高階 GP とは

例えば


```text
観測入力: [n, d]
観測応答: [n, h, w]
```

なら、各入力に対して `h x w` の画像を予測します。

1次元スペクトルも利用できますが、特に

- 画像
- 空間grid
- 時間 × 空間
- 波長 × 時間

のような複数出力軸を持つ tensor で自然です。

---

## 11.8 HOGP の covariance

入力軸と各出力軸に covariance を持たせます。

例えば出力が2次元なら概念的に


$$
K
=
K_X
\otimes
K_1
\otimes
K_2
$$

です。

ここで

- $`K_X`$: input covariance
- $`K_1`$: output axis 1 covariance
- $`K_2`$: output axis 2 covariance

です。

出力が3次元ならさらに


$$
K_3
$$

が加わります。

---

## 11.8.1 Mixed 高階 GP

---

## 11.9 なぜ Kronecker が効くのか

例えば出力 shape が


```text
10 × 20 × 30
```

なら出力要素数は


$$
10\times20\times30=6000
$$

です。

直接6000次元の covariance を持てば


$$
6000^2=36,000,000
$$

要素です。

一方 Kronecker 構造なら


```text
10×10
20×20
30×30
```

の小さな covariance の組み合わせとして表現できます。

これが structured covariance の主要な利点です。

---

## 11.10 HOGP の latent representation

HOGPは出力軸ごとの潜在座標上に共分散を構成します。潜在座標を学習する場合、
出力軸の関係も観測から推定されます。

---

## 11.11 `num_latent_dims`

`num_latent_dims` は各 output dimension を表す latent coordinate の次元数を制御します。

概念的には


```text
output axis
  ↓
latent coordinate
  ↓
kernel
  ↓
axis covariance
```

です。

latent dimensionを増やすとrepresentationの自由度は増えますが、必ず予測性能が
改善するわけではありません。parameter数や識別の難しさも増えるため、検証が必要です。

---

## 11.12 `learn_latent_pars`

出力軸の潜在座標を推定する場合と、既知の物理座標を直接使う場合では、
共分散に課す仮定と解釈が異なります。

---

## 11.13 HOGP の入力 shape

入力 $X$ の各観測点に、画像やスペクトルなどの構造化応答 $Y$ が対応します。
$Y$ の出力軸を独立の構造として保持します。

---

## 11.15 HOGP の fitting

構造化共分散の学習では、Kronecker構造を活用する線形代数と数値安定性が重要です。

---

## 11.16 HOGP posterior

予測分布は入力点と構造化出力軸の結合分布を表します。

---

## 11.17 HOGP posterior sampling

HOGP の posterior sampling は structured covariance を利用します。

概念的には


```text
prior sample
   +
conditioning correction
   ↓
posterior sample
```

です。

巨大な full posterior covariance を単純に構築するより、構造を活かしてsampleを得られます。

---

## 11.18 HOGP の仮定

HOGP は便利ですが、出力軸 covariance が Kronecker 分解できるという構造を利用しています。

したがって


```text
axis 1 と axis 2 の相互作用が
単純な separable structure では表現しにくい
```

場合にはモデルミスマッチが生じます。

Kronecker は計算効率と引き換えに structural assumption を置いていると理解する必要があります。

---

# 潜在座標の Kronecker GP

## 11.19 潜在座標の Kronecker GP とは

潜在座標の Kronecker GP は


```text
X: 実験条件・設計変数
T: 時間・波長・位置などの出力座標
Y: X × T 上の観測
```

を明示的に扱います。

モデル化のイメージは


$$
y = f(x,t)
$$

です。

covariance は概念的に


$$
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
$$

です。

---

## 11.20 `train_T`

Latent Kroneckerモデルでは、設計入力 $X$ と出力座標 $T$ を区別します。
例えば、材料組成が $X$、波長が $T$、測定スペクトルが $Y$ です。

---

## 11.21 HOGP と異なる点

HOGP では output tensor の index / latent coordinates が中心です。

潜在座標の Kronecker GP では


```text
波長 = 450 nm
時間 = 12.5 s
位置 = 3.2 mm
```

のような意味のある output coordinate を直接与えられます。

そのため


```text
Tに物理的意味がある
```

場合は非常に自然です。

---

## 11.22 新しい T で予測できる

出力座標上の共分散を用いるため、観測時と異なる出力座標で予測できます。
これは出力座標方向の補間や外挿に相当します。

---

## 11.23 missing output への対応

LatentKroneckerGPは、product-space structureを保ちながらmissing outputを扱うための
仕組みを持ちます。

つまり、すべての `X` で全 `T` が観測されていなくても扱える設計です。


```text
X1: T1 T2 T3 T4 観測
X2: T1 -- T3 T4 観測
X3: -- T2 T3 -- 観測
```

のような missing entries を想定できます。

これは完全gridを要求する単純な Kronecker model に対する重要な拡張です。

---

## 11.24 latent Kronecker structure

missing entries があると full Kronecker product を単純には使えません。

潜在座標の Kronecker GP は underlying latent product-space structure を維持しながら、観測された部分だけに conditioning します。


```text
latent full grid
  X × T
   ↓
Kronecker prior
   ↓
observed entriesのみconditioning
```

という考え方です。

---

## 11.25 pathwise conditioning

潜在座標の Kronecker GP は posterior sampling に pathwise conditioning を利用します。

概念的には


```text
1. prior function sample を生成
2. 観測とのずれを計算
3. conditioning correction を加える
4. posterior sample を得る
```

です。

posterior covariance を巨大な dense matrix として完全に構築する方法とは異なります。

---

## 11.26 iterative linear solvers

構造化共分散の推論では、大規模線形系の反復解法が必要になる場合があります。
反復法の収束精度は予測の数値誤差に影響します。

---

## 11.27 fitting example

Latent Kroneckerモデルの学習は、入力軸と出力座標軸の共分散パラメータを推定する問題です。

---

## 11.28 posterior example

新しい入力点と出力座標に対する予測では、両軸の共分散を用います。

---

## 11.30 covariance modules

入力軸と出力座標軸には、それぞれ異なる共分散関数を設定できます。

---

## 11.31 mean modules

入力と出力座標の平均構造は、それぞれの軸で異なるトレンドを表現できます。

---

## 11.32 高階 GP vs 潜在座標の Kronecker GP

---

## 11.33 選択フロー


```text
1回の実験からstructured responseが得られる
        ↓
出力軸に物理的な座標 T がある？
  ├─ Yes
  │    ↓
  │   新しいTで補間したい / 欠損Tがある？
  │    ├─ Yes → 潜在座標の Kronecker GP
  │    └─ No  → 潜在座標の Kronecker GP または HOGP
  │
  └─ No
       ↓
       tensor shape 自体が重要？
        ├─ Yes → 高階 GP
        └─ No  → 通常の multi-output GP も検討
```

---

## 11.34 Multi-output GP との違い

Multi-output GP は


```text
y1, y2, ..., ym
```

を複数outputとして扱います。

Structured Output GP はそれらの output が


```text
規則的なgrid
波長軸
時間軸
空間座標
```

を持つことを利用します。

したがって単に output 数が多いだけなら HOGP / 潜在座標の Kronecker GP が必ずしも必要ではありません。

---

## 11.35 Kronecker 構造のマルチタスク GP との違い

Kronecker 構造のマルチタスク GP も Kronecker structure を利用します。

ただし主な意味は


```text
X × task
```

です。

一方 Structured Output では


```text
X × output coordinate
```

または


```text
X × tensor axes
```

です。

時間や波長を単なる task label として扱うより、連続座標として扱う方が自然な場合があります。

---

## 11.36 独立モデルの集合 との違い

出力位置ごとに独立したモデルを学習する場合、出力位置間の相関は共有されません。
構造化出力GPでは共分散を通じてその依存関係を表現できます。

---

## 11.37 structured response をそのまま最大化するわけではない

Bayesian Optimization では最終的に「何を最大化 / 最小化するのか」を決める必要があります。

structured output


$$
y(x,t)
$$

をそのまま scalar acquisition に渡すのではなく、通常は objective を定義します。

例えばスペクトルなら

- target wavelength の強度
- peak height
- peak position
- 積分面積
- target spectrum との距離
- 複数bandの比

などです。

---

## 11.38 objective function の例

構造化応答 $Y(x)$ に対する効用の例として、特定座標 $t_0$ の応答
$U_1(x)=Y(x,t_0)$、出力座標に沿う積分 $U_2(x)=\int Y(x,t)\,dt$、
目標曲線 $y_\mathrm{target}$ からの距離
$U_3(x)=-\int (Y(x,t)-y_\mathrm{target}(t))^2\,dt$ が考えられます。

---

## 11.39 posterior sample 上で objective を評価する

structured posterior から sample


$$
Y^{(s)}(x)
$$

を取り、各sampleに objective


$$
g(Y^{(s)}(x))
$$

を適用すれば、structured uncertainty を scalar objective uncertainty に伝播できます。


```text
structured posterior
      ↓ sample
spectrum / image sample
      ↓ objective
scalar score sample
      ↓
MC acquisition
```

です。

---

## 11.40 non-linear objective で特に有効

例えば


```text
maximum peak
peak position
threshold exceedance area
spectrum similarity
```

のような nonlinear functional は posterior mean にだけ適用するより、posterior sample ごとに評価する方が uncertainty
を自然に反映できます。

---

## 11.41 Active Learning

目的が最適化ではなく structured response 全体の精度向上なら Active Learning として考えます。

例えば


```text
全波長の予測誤差を減らしたい
画像全体のuncertaintyを減らしたい
未知時間領域を学びたい
```

という目的です。

この場合は integrated variance や information gain のような基準が自然です。

---

# 実務上の注意

## 11.42 output normalization

structured output は位置によってscaleが異なることがあります。

例えばスペクトルで


```text
一部band: 0〜1
別band: 0〜1000
```

なら、scaleの大きい領域が fitting を支配する可能性があります。

物理的意味を壊さない範囲で scale を揃えることを検討します。

---

## 11.43 input normalization

通常のGPと同様、`X` は unit cube 近辺にnormalizeするのが基本です。

潜在座標の Kronecker GPでは `T` のscaleもkernel lengthscaleと関係するため、


```text
時間: 0〜100000
波長: 400〜800
```

のような場合は適切なスケーリングを検討します。

---

## 11.44 出力gridが大きすぎる場合

Kronecker structure を使っても、出力gridが極端に大きいと計算負荷は大きくなります。

例えば高解像度画像なら

- downsampling
- PCA / latent representation
- autoencoder
- DKL
- patch-based model

なども候補です。

Structured GP は「構造化すれば無限に大きい出力を扱える」わけではありません。

---

## 11.45 PCAとの違い

PCAで output を低次元latent vectorに圧縮してGPを作る方法もあります。


```text
Y
 ↓ PCA
z
 ↓ GP
z_hat
 ↓ inverse PCA
Y_hat
```

これは有力なbaselineです。

一方 HOGP / 潜在座標の Kronecker GP は output structure を GP covariance の中で直接扱います。

---

## 11.46 PCAが向く場合

- output dimension が非常に大きい
- 少数principal componentsで十分説明できる
- grid geometryよりglobal latent variationが重要

なら PCA + GP は有力です。

---

## 11.47 HOGPが向く場合

- tensor axes自体が重要
- axisごとのseparable correlationが期待できる
- 画像やgrid responseを直接扱いたい
- Kronecker structureが妥当

なら HOGP が候補です。

---

## 11.48 LatentKroneckerGPが向く場合

- output coordinate `T` に物理的意味がある
- `T` 上でinterpolationしたい
- missing output entries がある
- スペクトル・時系列・空間profile
- product-space covarianceが自然

なら 潜在座標の Kronecker GP が候補です。

---

## 11.49 separability の診断

Kronecker model の大きな仮定は separability です。

例えば


$$
k((x,t),(x',t'))=k_X(x,x')k_T(t,t')
$$

が妥当かを考えます。

もし


```text
あるX領域ではT方向が非常に滑らか
別X領域ではT方向が急激
```

なら単純なseparable kernelでは不足する可能性があります。

---

## 11.50 non-stationarity

スペクトルや時系列では局所的に異なるsmoothnessを持つ場合があります。

例えば


```text
baseline領域: smooth
phase transition周辺: sharp
```

です。

stationary RBF / Matérn kernel だけでは表現しづらい場合、kernel拡張やinput warpingを検討します。

---

## 11.51 noise structure

structured output では noise も位置依存する場合があります。


```text
特定波長だけS/Nが悪い
時間後半だけ測定誤差が大きい
画像端部でnoiseが増える
```

標準 Gaussian likelihood が仮定する単純noiseでは不足する可能性があります。

---

## 11.52 missingness mechanism

潜在座標の Kronecker GP が missing entries を扱えるとしても、missingnessが informative なら注意が必要です。

例えば


```text
測定不能だった箇所だけmissing
```

なら missingness 自体が応答に依存している可能性があります。

単に欠損を無視するだけではbiasが生じます。

---

## 11.53 extrapolation in T

潜在座標の Kronecker GP は新しい `T` でposteriorを評価できますが、interpolationとextrapolationは別です。


```text
train T: 400〜800 nm
predict T: 450 nm
  → interpolation

predict T: 1200 nm
  → extrapolation
```

後者はkernel priorへの依存が大きくなります。

---

## 11.54 diagnostic plots

structured output model では scalar RMSE だけでなく、構造ごとの診断が重要です。

例えば

- wavelength-wise RMSE
- time-wise RMSE
- image residual map
- peak position error
- integrated area error
- uncertainty calibration by coordinate

です。

---

## 11.55 holdout design

random element holdout だけでなく、目的に応じて holdout を変えます。


```text
new X prediction
  → Xごとholdout

new T interpolation
  → T座標をholdout

missing-grid robustness
  → X×T entryをmask
```

です。

---

## 11.57 高階 GP example

Higher-order Gaussian process の数値例は実装ガイドを参照してください。

---

## 11.58 潜在座標の Kronecker GP example

Latent Kroneckerモデルの数値例は実装ガイドを参照してください。

---

## 11.59 raw data

元の観測データと、推論に使う内部表現は異なる場合があります。

---

### 「出力数が多ければ HOGP」

必ずしもそうではありません。出力に規則的なtensor structureがあるかが重要です。

### 「潜在座標の Kronecker GP の T は task ID」

必ずしもtask labelではありません。時間・波長・位置など連続座標として使えます。

### 「Kronecker を使えばどんな巨大データでも軽い」

違います。構造による効率化はありますが、grid sizeやtraining sizeが増えれば計算負荷も増えます。

### 「structured outputをそのままEIに入れればよい」

通常はstructured responseからscalar / low-dimensional objectiveを定義します。

### 「新しいTで予測できるなら外挿も安全」

違います。training range外ではpriorへの依存が大きくなります。

---

## 11.61 出力構造の比較表

| 出力構造 | 対応する考え方 |
|---|---|
| scalar response | scalar GP |
| 少数の独立outputs | independent / 独立モデルの集合 |
| related task labels | multi-task GP |
| tensor axesそのものに構造がある | 高階 GP |
| 明示的なoutput coordinate `T` がある | 潜在座標の Kronecker GP |
| mixed design X + tensor output | 混合入力 Higher Order GP |
| mixed design X + explicit `T` | 混合入力 Latent Kronecker GP |
| 非常に巨大で低rankなoutput | reduction / latent modelも検討 |

---

## 11.62 数学的な全体像

Structured Output GP の基本は


$$
f(x,t)\sim\mathcal{GP}(m,k)
$$

と


$$
k((x,t),(x',t'))
=
k_X(x,x')k_T(t,t')
$$

です。

grid全体では


$$
K
=
K_X\otimes K_T
$$

となります。

HOGP では複数 output axes に拡張して


$$
K
=
K_X
\otimes K_1
\otimes \cdots
\otimes K_p
$$

を利用します。

---

## 11.63 BOとの全体像


```text
X
 ↓
structured-output surrogate
 ↓
Y(x, T) posterior
 ↓ posterior sampling
structured response samples
 ↓ objective g(Y)
scalar utility samples
 ↓
acquisition function
 ↓
next X
```

です。

この分解を理解すると、model と objective と acquisition の責任範囲が明確になります。

---

## 11.64 この章で覚えておくこと

Structured Output GP は、出力を単なる多数のscalar valuesとしてではなく、**出力側の構造を利用してモデル化するGP**です。

高階 GP / 潜在座標の Kronecker GP

を使います。

重要点は次の通りです。

1. HOGP は tensor-valued output の各軸に Kronecker structure を利用する
2. 潜在座標の Kronecker GP は `X × T` のproduct-space model
3. `T` は時間・波長・位置などの明示的なoutput coordinateとして使える
4. 潜在座標の Kronecker GP は新しい `T` でposteriorを評価できる
5. 潜在座標の Kronecker GP はmissing output entriesを意識した設計
6. HOGP fittingでは Kronecker / linear-operator structure を壊さない solver と optimizer の選択が重要
7. 潜在座標の Kronecker GP では iterative methods が重要
8. Kronecker efficiency は separable covariance assumption と引き換え
9. BOではstructured responseから何を最適化するかobjectiveを明示する
10. posterior sample上でobjectiveを評価するとstructured uncertaintyを伝播できる

次章では、条件付き探索空間・異種task・contextを扱う [Hierarchical / Contextual GP](12_hierarchical_contextual_gp.md)
を説明します。

## 参考文献

1. Zhe, S., Xing, W., and Kirby, R. M. (2019).
   Scalable high-order Gaussian process regression.
   *Proceedings of AISTATS*.
2. BoTorch 高階 GP and 潜在座標の Kronecker GP documentation.
3. [Gaussian Process](02_gaussian_process.md)
4. [Kernel](03_kernel.md)
5. [Multi-task / Multi-output](07_multitask_multioutput.md)

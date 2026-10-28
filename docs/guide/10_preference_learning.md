# Preference Learning：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/10_preference_learning.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

数値スコアではなく候補点の対比較（どちらが好ましいか）から潜在utilityを推定する。単一目的の数値回帰における`train_Y`と比較indexを同一視しない。対比較の向き、比較の重複統合、観測尤度の仮定を確認する。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/preference/pairwise.py`の`PairwiseGP`はBoTorch `PairwiseGP`を継承し、元のdatapoints/comparisonsをraw snapshotsとして保存する。
- **Delegated:** Laplace近似、重複の統合、posterior、pairwise likelihoodはBoTorch。
- **Implemented（接続）:** `make_mll()`は`PairwiseLaplaceMarginalLogLikelihood`を返す。標準GPの`ExactMarginalLogLikelihood`ではない。

## 最小の利用例

```python
import torch
from robotorchan.models import PairwiseGP

points = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
comparisons = torch.tensor([[1, 0], [1, 2]], dtype=torch.long)
model = PairwiseGP(datapoints=points, comparisons=comparisons)
print(type(model.make_mll()).__name__)
```

コードは確認した公開APIを示す最小例であり、すべての条件での成功を保証するものではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

比較は`(勝者index, 敗者index)`の契約で扱う。比較グラフの連結性、正則化、Laplace approximationと潜在utilityの尺度を区別する。静的API検査を「全ての比較集合で学習成功」と読み替えない。対応する既存テスト・Notebookの正確な参照は移行時に照合する。

[旧Model Mapping](../../.github/project-records/theory-guide/mapping/models.md)および[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../models/README.md)を参照する。

## Pairwise likelihoodとLaplace MLLの実装上の違い

候補xiとxjの潜在utility u(x)について、比較データがiをjより好むと表す場合は、確率的にP(i≻j|u_i,u_j)をlikelihoodで定義する。比較の行に観測されるのは連続値y_iではなく、候補のindexの組である。

src/robotorchan/models/preference/pairwise.pyのPairwiseGPはBoTorch PairwiseGPを継承し、datapoints・comparisons・pairwise likelihood・入力変換を親へ渡す。Laplace approximation、重複点のconsolidation、utility posteriorの演算はBoTorchの担当となり、robotorchanはconstructorのraw_datapoints/raw_comparisonsをdetach/cloneしたsnapshotとして保持する。

make_mll()はPairwiseLaplaceMarginalLogLikelihood(self.likelihood,self)を返す。ExactGPModelMixinのExactMarginalLogLikelihoodでも、VariationalELBOでもない。例えばcomparisons=[[1,0],[1,2]]は候補1が0と2に優越する観測を表すが、utilityの絶対尺度・ランキングの全順序・データの比較整合性を自動保証しない。

priorとlikelihoodの指定、comparison indexの有効範囲、モデルの数値条件、Laplace近似の収束は個別に検査すべき事項である。make_mll()が存在することはfit()の共通実装が存在することの証拠ではない。

**検証境界：** ソースのconstructor・make_mll・raw保持を照合。比較グラフに対するposteriorの数学的整合、収束、選好予測性能はこの説明だけでは独立数値テストの完了を保証しない。

## 実装補足：理論本文から分離した対応情報

### 10.20 robotorchan の `PairwiseGP`

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

---


### 10.43 raw data API

比較学習では、入力候補と勝者・敗者の関係が教師情報です。
通常の実数値回帰ターゲットと同一視しません。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

---


### 10.70 robotorchan Notebook

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

---


### 10.77 robotorchan で覚えておく API

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

---

## 理論章から分離した実装上の条件

### 10.15 PairwiseGP の default kernel

BoTorch の `PairwiseGP` は default で scaled RBF kernel を使います。

```text
RBF kernel
  → input similarity

ScaleKernel
  → latent utility scale
```

probit preference model では observation-noise scale と utility scale の識別が通常回帰ほど単純ではないため、BoTorch は
kernel output scale を使う設計です。

---

#### 10.1 この章で扱うこと：実装上の補足

robotorchan では BoTorch の

#### 10.1 この章で扱うこと：実装上の補足

を BoTorch-compatible wrapper として提供しています。

#### 10.1 この章で扱うこと：実装上の補足

- preference data
- latent utility
- pairwise comparison likelihood
- probit model
- GP prior
- Laplace approximation
- `PairwiseLaplaceMarginalLogLikelihood`
- posterior utility
- pairwise BO
- Expected Utility of Best Option (EUBO)
- BOPE との関係
- robotorchan API

#### 10.7 Probit likelihood：実装上の補足

BoTorch の `PairwiseGP` は default で **probit likelihood** を使います。

#### 10.7 Probit likelihood：実装上の補足

Chu & Ghahramani の定式化では denominator にノイズscale $`\sigma`$ を含めますが、BoTorch はこの scale を明示的に
likelihood parameter として分離せず、function scale を kernel の `ScaleKernel` で表現します。

#### 10.17 relative scale と評価指標：実装上の補足

BoTorch の preference tutorial でも、latent value の絶対値ではなく ranking correlation を評価に利用しています。

#### 10.21 constructor：実装上の補足

### 10.21 constructor

#### 10.22 `make_mll()`：実装上の補足

### 10.22 `make_mll()`

#### 10.27 observation noise の扱い：実装上の補足

通常の regression GP では `posterior(..., observation_noise=True)` で observation noise を含む predictive
distribution を考えます。

#### 10.38 q=2 の直感：実装上の補足


BoTorch の preference BO tutorial では `q=2` として EUBO を最適化する例があります。

#### 10.53 BOPE：実装上の補足

BoTorch には Bayesian Optimization with Preference Exploration、つまり **BOPE** の考え方があります。

#### 10.60 comparison noise と scale：実装上の補足

BoTorch は likelihood 内で明示的な $`\sigma`$ を持たせる代わりに、kernel output scale 側で function scale を扱います。

#### 10.80 この章で覚えておくこと：実装上の補足

robotorchan では

## 数値 API と利用上の補足

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

一方 `pairwise GP` の default は probit です。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

具体的な API と実行例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

実装例は [Preference Learning implementation](../models/preference_learning.md) を参照してください。

1. `comparisons` は `[winner, loser]` の index pair
2. latent utility `f(x)` に GP prior を置く
3. default likelihood は probit
4. non-Gaussian posterior を Laplace approximation する
5. `make_mll()` は `Laplace 近似対数周辺尤度`
6. utility の absolute value より ranking が重要
7. preference BO では EI より preference-specific acquisition を考える
8. BoTorch では EUBO が利用できる
9. multiple outcomes と preference を分ける場合は BOPE が有力
10. evaluator heterogeneity や cyclic preference が強い場合は単一 utility 仮定を見直す

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

`train_X` / `train_Y`

一方 `PairwiseGP` の default は probit です。

```text
Bradley–Terry / logistic
  → sigmoid

PairwiseGP default
  → Gaussian CDF / probit
```


そこで `PairwiseGP` は **Laplace approximation** を使います。

このため `PairwiseGP` は名前に GP とありますが、標準 `SingleTaskGP` と posterior fitting の仕組みが異なります。

`PairwiseGP` 内部では preference comparisons と GP prior の両方を考慮して latent utility の MAP 解を求めます。

PairwiseGP では fitting を理解するとき、2段階を区別するとよいです。

PairwiseGP の probit preference model では noise と utility scale の識別が異なるため、通常回帰と同じ意味で
observation-noise variance を解釈しません。


さらに PairwiseGP は pair 自体ではなく、各候補に latent utility function `f(x)` を置いています。

ここまでの `PairwiseGP` は、**比較データからutility posteriorを作るsurrogate model**です。
これだけでは「次にどの2候補を比較するか」は決まりません。

~~~text
PairwiseGP
  → preference surrogate

Preference acquisition
  → 次に比較する候補を評価

Acquisition optimization
  → 実際のquery pairを決定
~~~


```text
候補 x_i, x_j を提示
  ↓
どちらが良いか比較を観測
  ↓
PairwiseGP
  ↓
latent utility posterior
  ↓
次の比較候補を選ぶ
```


```text
PairwiseGP posterior
      ↓
EUBO
      ↓
2候補を選ぶ
      ↓
比較結果を取得
      ↓
comparisonsに追加
```


`PairwiseGP` は `condition_on_observations` をサポートしますが、通常回帰モデルと違い、新しい `Y` も pairwise comparisons
である必要があります。

PairwiseGP の kernel hyperparameters も preference data に基づいて学習されます。

単一 `PairwiseGP` は基本的に1つの latent utility function を想定するため、個人差が大きい場合は

PairwiseGP を無理に当てはめないことが重要です。


を直接 PairwiseGP で学習します。

このとき PairwiseGP で expert utility を学習することができます。

```text
毎回、正確な強度値が測れる
→ SingleTaskGP

最終評価が人の総合判断で数値化しにくい
→ PairwiseGP を検討
```


## 10.58 PairwiseGP の計算コスト

PairwiseGP では latent utility MAP と Laplace approximation が必要なため、standard exact GP regression より
fitting が重くなる場合があります。

PairwiseGP は point ranking だけでなく posterior uncertainty を持ちます。

## 10.69 PairwiseGP と classifier の使い分け

なら latent utility model である PairwiseGP が自然です。

### 「PairwiseGP は binary classifier」

### 「PairwiseGP は Exact Gaussian regression」

```text
評価対象がある
   ↓
信頼できる absolute scalar score を取得できる？
 ├─ Yes
 │    └─ SingleTaskGP 等
 │
 └─ No
      ↓
      pairwise comparison なら安定して得られる？
       ├─ Yes
       │    └─ PairwiseGP
       │
       └─ No
            └─ 別の preference / ordinal model を検討
```


| 観測・仮定 | 対応する考え方 |
|---|---|
| scalar targetを直接観測 | regression GP |
| pairwise comparisonを観測 | `PairwiseGP` |
| 複数outcomeは観測でき、utilityを比較で学ぶ | BOPE型の分離 |
| ordinal scoreを直接観測 | ordinal model |
| evaluator差を明示的に表現したい | hierarchical / multitask preference model |
| 強いcyclic preferenceがある | scalar utility仮定を再検討 |

```text
pairwise comparison
      ↓
PairwiseGP
      ↓
latent utility posterior
      ↓
EUBO / preference acquisition
```


1. `comparisons` は `[winner, loser]` の index pair
2. latent utility `f(x)` に GP prior を置く
3. default likelihood は probit
4. non-Gaussian posterior を Laplace approximation する
5. `make_mll()` は `PairwiseLaplaceMarginalLogLikelihood`
6. utility の absolute value より ranking が重要
7. preference BO では EI より preference-specific acquisition を考える
8. BoTorch では EUBO が利用できる
9. multiple outcomes と preference を分ける場合は BOPE が有力
10. evaluator heterogeneity や cyclic preference が強い場合は単一 utility 仮定を見直す

1. Chu, W., and Ghahramani, Z. (2005).
   Preference learning with Gaussian processes.
   *Proceedings of the 22nd International Conference on Machine Learning*, 137-144.
2. Lin, Z. J. et al. (2022).
   Preference exploration for efficient Bayesian optimization with multiple outcomes.
   *Proceedings of AISTATS*.
3. Astudillo, R. et al. (2023).
   qEUBO: A decision-theoretic acquisition function for preferential Bayesian optimization.
   *Proceedings of AISTATS*.
4. BoTorch `PairwiseGP`, preference BO, and BOPE documentation.
5. [Gaussian Process](../theory/02_gaussian_process.md)
6. [Acquisition Function](../theory/04_acquisition_function.md)

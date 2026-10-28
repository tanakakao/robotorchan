# Structured Output / Higher-Order GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../theory/11_structured_output.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

応答がスカラーではなく画像・曲線・テンソルなどの構造を持つ場合、出力要素間の共分散を利用する。構造化出力のjoint posteriorと、単一のスカラーobjectiveを扱う通常BOの獲得関数との間には出力変換・shape契約が必要となる。

## 実装の責務と公開契約

- **Extended:** `src/robotorchan/models/structured/higher_order.py`の`HigherOrderGP`はBoTorch `HigherOrderGP`を継承しraw入力・tensor出力の保持と`make_mll()`を加える。
- **Delegated:** 出力軸の潜在パラメータ、tensor covariance、Kronecker構造の線形代数とposteriorはBoTorch。
- **Extended:** `MixedHigherOrderGP`はdesign-inputの共分散因子のみnative mixed kernelへ置き換える。現在はカスタム`covar_modules`指定を拒否する。
- ほかに`src/robotorchan/models/structured/latent_kronecker.py`の別familyがあり、同一アルゴリズムとみなさない。

## 最小の利用例

```python
import torch
from robotorchan.models import HigherOrderGP

X = torch.tensor([[0.1], [0.4], [0.8]], dtype=torch.double)
Y = torch.rand(3, 2, 2, dtype=torch.double)
model = HigherOrderGP(train_X=X, train_Y=Y)
print(type(model).__name__)
```

コードは確認した公開APIを示す最小例であり、すべての条件での成功を保証するものではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

上の例はモデル構築までの使用形態であり、tensor-output posteriorの完全なshape/数値検証や学習成功を宣言するものではない。tensor出力を単純に潰して単一出力posteriorとして扱わない。関連：[Structured Outputモデル](../models/structured_output.md)。

[旧Model Mapping](../../.github/project-records/theory-guide/mapping/models.md)および[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../models/README.md)を参照する。

## Tensor出力における共分散因子と出力座標

入力Xを[n,d]、tensor出力Yを[n,r1,...,rk]とするHigher-Order GPでは、入力側のcovarianceと出力各軸のcovarianceを因子として扱う。概念的な分離可能共分散は、

$$
\operatorname{Cov}(f(x,a_1,\ldots,a_k),f(x',a'_1,\ldots,a'_k))
=k_X(x,x')\prod_{j=1}^{k} k_j(a_j,a'_j)
$$

の形で表せる。これはfactorized priorの説明であり、全てのstructured-outputモデルが同じkernel因子やposterior構造を持つという主張ではない。

src/robotorchan/models/structured/higher_order.pyのHigherOrderGPはBoTorchHigherOrderGPを継承し、covar_modules・num_latent_dims・learn_latent_pars・latent_initなどを親へ委譲する。robotorchanはraw_train_X/raw_train_Yとraw_train_Yvar=Noneのsnapshot、ExactGPModelMixin.make_mll()を追加する。構造出力のpredictive covariance、Kronecker処理およびposteriorサンプリングはBoTorch側の責務となる。

MixedHigherOrderGPはdesign入力だけをカテゴリ＋連続のmixed kernelへ変更する。src/robotorchan/models/structured/higher_order.pyでcovar_modules[0]をmake_mixed_covar_moduleで置き換え、出力軸側のfactorを保持する。custom covar_modulesを渡す場合はValueErrorで拒否し、cat_dimsは元の設計入力列の番号を取る。

**注意：** posteriorが多次元tensor出力を表す場合、tensorを単純にflattenしてscalar acquisitionへ渡しても意味論的な目的の定義にはならない。最大・積分・特定画素などのfunctionalを別に定義する必要がある。今回の根拠はconstructorとkernel差し替え処理のソース確認であり、tensor posteriorの数値的同値性・学習収束は未検証である。

## 実装補足：理論本文から分離した対応情報

### 11.14 robotorchan の HOGP wrapper

Higher-order Gaussian process は入力と出力軸の共分散構造を分離して扱います。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

---


### 11.29 robotorchan の raw data API

観測時の入力・出力座標・応答と、推論時に内部変換された表現を区別します。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

---


### 11.56 robotorchan Notebook

理論を用いた数値例は実装ガイドのNotebook案内を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

---

## 理論章から分離した実装上の条件

#### 11.1 この章で扱うこと：実装上の補足

robotorchan では主に

#### 11.1 この章で扱うこと：実装上の補足

- structured output とは何か
- vector output と tensor output の違い
- separable covariance
- Kronecker product
- `HigherOrderGP`
- latent output coordinates
- `LatentKroneckerGP`
- missing outputs
- pathwise conditioning
- iterative inference
- structured output と Bayesian Optimization
- robotorchan での使い分け

#### 11.6 HOGP と LatentKroneckerGP の違い：実装上の補足

robotorchan で最も重要な区別です。

#### 11.7 HigherOrderGP とは：実装上の補足

BoTorch の `HigherOrderGP` は tensor-valued output のための GP です。

#### 11.7 HigherOrderGP とは：実装上の補足

BoTorch の説明でいう “higher-order” は、予測対象が matrix / tensor であることを意味します。

#### 11.8.1 Mixed HigherOrderGP：実装上の補足

現在の robotorchan には `MixedHigherOrderGP` があります。

#### 11.8.1 Mixed HigherOrderGP：実装上の補足

HOGP の Kronecker factorization は design-input factor と tensor-output axes の factor を分離します。
Mixed版はこのうち **design-input covarianceだけ**を robotorchan の mixed covariance に置き換え、
tensor-output側の Kronecker structure は維持します。

#### 11.8.1 Mixed HigherOrderGP：実装上の補足

したがって `cat_dims` は `train_X` の categorical design columns を表し、
output-axis coordinate の指定ではありません。

#### 11.17 HOGP posterior sampling：実装上の補足

BoTorchのHOGP posteriorはstructured sampling pathを利用します。
Matheron型のconditioningは、その構造を保ったposterior samplingを理解するための
重要な考え方です。

#### 11.42 output normalization：実装上の補足

BoTorch の HOGP は structured output 用の outcome transform を持ち、default transformも利用します。

#### 11.64 この章で覚えておくこと：実装上の補足

robotorchan では主に

## 数値 API と利用上の補足

また現在の Mixed wrapper は design-input covariance を内部管理するため、
custom `covar_modules` との同時指定はサポートしません。
これは HOGP 理論上の制約ではなく、現在の public implementation contract です。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

実装例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

具体的なAPI・実行例は [Structured Output implementation](../models/structured_output.md) を参照してください。

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

`HigherOrderGP` / `LatentKroneckerGP`

```text
HigherOrderGP
  └─ 出力 tensor の shape / 各軸の構造を利用する

LatentKroneckerGP
  └─ 入力 X と明示的な出力座標 T の積空間をモデル化する
```


# HigherOrderGP

```text
train_X: [n, d]
train_Y: [n, h, w]
```


# LatentKroneckerGP

## 11.19 LatentKroneckerGP とは

`LatentKroneckerGP` は

LatentKroneckerGP では

LatentKroneckerGP は underlying latent product-space structure を維持しながら、観測された部分だけに conditioning します。

LatentKroneckerGP は posterior sampling に pathwise conditioning を利用します。

## 11.32 HigherOrderGP vs LatentKroneckerGP

| 観点 | HigherOrderGP | LatentKroneckerGP |
|---|---|---|
| 主用途 | tensor-valued output | X × T product-space output |
| output coordinate | latent / tensor axis | 明示的 `T` |
| 画像 | 得意 | T設計次第 |
| スペクトル | 可能 | 特に自然 |
| 時系列 | 可能 | 特に自然 |
| 新しい出力座標 | 基本的に軸shape依存 | `test_T` で自然 |
| missing outputs | 構造上制約あり | missing entries を意識した設計 |
| posterior sampling | Matheron's rule | pathwise conditioning |
| fitting | `_fast_solves` + torch optimizer | iterative methods |
| raw固有API | `raw_train_X`, `raw_train_Y` | `raw_train_X`, `raw_train_T`, `raw_train_Y` |

```text
1回の実験からstructured responseが得られる
        ↓
出力軸に物理的な座標 T がある？
  ├─ Yes
  │    ↓
  │   新しいTで補間したい / 欠損Tがある？
  │    ├─ Yes → LatentKroneckerGP
  │    └─ No  → LatentKroneckerGP または HOGP
  │
  └─ No
       ↓
       tensor shape 自体が重要？
        ├─ Yes → HigherOrderGP
        └─ No  → 通常の multi-output GP も検討
```


したがって単に output 数が多いだけなら HOGP / LatentKroneckerGP が必ずしも必要ではありません。

## 11.35 KroneckerMultiTaskGP との違い

`KroneckerMultiTaskGP` も Kronecker structure を利用します。

## 11.36 ModelListGP との違い

`LatentKroneckerGP`では `T` のscaleもkernel lengthscaleと関係するため、

一方 HOGP / LatentKroneckerGP は output structure を GP covariance の中で直接扱います。

なら LatentKroneckerGP が候補です。

LatentKroneckerGP が missing entries を扱えるとしても、missingnessが informative なら注意が必要です。

`LatentKroneckerGP` は新しい `T` でposteriorを評価できますが、interpolationとextrapolationは別です。

## 11.57 HigherOrderGP example

## 11.58 LatentKroneckerGP example

### 「LatentKroneckerGP の T は task ID」

| 出力構造 | 対応する考え方 |
|---|---|
| scalar response | scalar GP |
| 少数の独立outputs | independent / `ModelListGP` |
| related task labels | multi-task GP |
| tensor axesそのものに構造がある | `HigherOrderGP` |
| 明示的なoutput coordinate `T` がある | `LatentKroneckerGP` |
| mixed design X + tensor output | `MixedHigherOrderGP` |
| mixed design X + explicit `T` | `MixedLatentKroneckerGP` |
| 非常に巨大で低rankなoutput | reduction / latent modelも検討 |

`HigherOrderGP` / `LatentKroneckerGP`

1. HOGP は tensor-valued output の各軸に Kronecker structure を利用する
2. LatentKroneckerGP は `X × T` のproduct-space model
3. `T` は時間・波長・位置などの明示的なoutput coordinateとして使える
4. LatentKroneckerGP は新しい `T` でposteriorを評価できる
5. LatentKroneckerGP はmissing output entriesを意識した設計
6. HOGP fittingでは Kronecker / linear-operator structure を壊さない solver と optimizer の選択が重要
7. LatentKroneckerGP では iterative methods が重要
8. Kronecker efficiency は separable covariance assumption と引き換え
9. BOではstructured responseから何を最適化するかobjectiveを明示する
10. posterior sample上でobjectiveを評価するとstructured uncertaintyを伝播できる

1. Zhe, S., Xing, W., and Kirby, R. M. (2019).
   Scalable high-order Gaussian process regression.
   *Proceedings of AISTATS*.
2. BoTorch `HigherOrderGP` and `LatentKroneckerGP` documentation.
3. [Gaussian Process](../theory/02_gaussian_process.md)
4. [Kernel](../theory/03_kernel.md)
5. [Multi-task / Multi-output](../theory/07_multitask_multioutput.md)

## 専用処理の補足

| 観点 | 高階 GP | 潜在座標の Kronecker GP |
|---|---|---|
| 主用途 | tensor-valued output | X × T product-space output |
| output coordinate | latent / tensor axis | 明示的 `T` |
| 画像 | 得意 | T設計次第 |
| スペクトル | 可能 | 特に自然 |
| 時系列 | 可能 | 特に自然 |
| 新しい出力座標 | 基本的に軸shape依存 | `test_T` で自然 |
| missing outputs | 構造上制約あり | missing entries を意識した設計 |
| posterior sampling | Matheron's rule | pathwise conditioning |
| fitting | `_fast_solves` + torch optimizer | iterative methods |
| raw固有API | `raw_観測入力`, `raw_観測応答` | `raw_観測入力`, `raw_train_T`, `raw_観測応答` |

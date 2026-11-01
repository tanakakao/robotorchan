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

torch.manual_seed(0)
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

## HigherOrderGPの出力軸と学習

`HigherOrderGP`は`train_X: [n,d]`と`train_Y: [n,r1,...,rk]`を受け取る。
例えば画像なら`[n,h,w]`であり、単なる出力ベクトルと違い、各軸に共分散構造を置く。
`num_latent_dims`、`learn_latent_pars`、`latent_init`は出力軸の潜在座標を制御する。
これは波長や時刻などの既知の物理座標をそのまま与えることとは異なる。

構造化出力用のoutcome transformとKroneckerの線形代数を維持して学習する。
以下は上の構築例に続けるHOGP用の学習経路である。

```python
from botorch.fit import fit_gpytorch_mll_torch
from linear_operator.settings import _fast_solves

with _fast_solves(True):
    fit_gpytorch_mll_torch(
        model.make_mll(),
        step_limit=25,
    )
posterior = model.posterior(X[:2])
assert posterior.mean.shape == (2, 2, 2)
```

HOGPのstructured posteriorは出力shapeを保ち、構造を使ったsampling pathを持つ。
Matheron型のconditioningはこのsamplingを理解するための基礎となる。
単に通常のExact GPと同じSciPy最適化器へ置き換える前に、
[構造化出力モデルガイド](../models/structured_output.md)の学習上の注意を確認する。

## LatentKroneckerGPの明示的な出力座標

`LatentKroneckerGP`は`X × T`の積空間をモデル化する。
例えば`train_X: [n,d]`、`train_T: [m,1]`、`train_Y: [n,m]`でスペクトルを表す。
`T`は波長・時刻・位置などの座標であり、単なるカテゴリtask IDと同じではない。
観測部分へのconditioning、反復線形ソルバ、pathwise posterior samplingはBoTorchへ委譲する。

```python
from botorch.fit import fit_gpytorch_mll
from robotorchan.models import LatentKroneckerGP

train_X = torch.linspace(0, 1, 5, dtype=torch.double).unsqueeze(-1)
train_T = torch.linspace(0, 1, 4, dtype=torch.double).unsqueeze(-1)
train_Y = torch.sin(train_X + train_T.transpose(-1, -2))
latent_model = LatentKroneckerGP(train_X, train_T, train_Y)
test_T = torch.tensor([[0.25], [0.75]], dtype=torch.double)
with latent_model.use_iterative_methods():
    fit_gpytorch_mll(
        latent_model.make_mll(),
        optimizer_kwargs={"options": {"maxiter": 25}},
    )
    prediction = latent_model.posterior(train_X[:2], test_T)
mean = prediction.mean.reshape(2, 2)
assert torch.isfinite(mean).all()
```

BoTorch 0.18.1のこのposteriorは、`mean`をXとTの積に沿う平坦な形で返す。
このbatchなしの例では要素数4の値を`[test_X数, test_T数]`へ戻している。
posterior sampleとmomentのshapeを同じと推測せず、batch次元も含めて確認する。

新しい`test_T`で予測できるが、補間と外挿の信頼性を区別する。
`T`の尺度はlengthscaleに関係するため、物理単位と変換を明示する。
欠損出力への対応は上流のmasking契約に従い、欠損自体が応答に依存する場合の
統計的な偏りまで自動的に解消するものではない。

入力と出力座標には別々に`covar_module_X` / `covar_module_T`、
`mean_module_X` / `mean_module_T`を指定できる。
`MixedLatentKroneckerGP`はX側だけをmixed共分散にする。
このモデルでも`cat_dims`は`train_X`の列であり、`train_T`の指定ではない。

## モデルを選ぶ基準

| 観点 | HigherOrderGP | LatentKroneckerGP |
| --- | --- | --- |
| 主な応答 | 画像などのtensor | 時刻・波長などの座標に沿った応答 |
| 出力側の表現 | 各tensor軸の潜在座標 | 明示的な`T` |
| 新しい出力位置 | 学習したtensor軸の構造に依存 | `test_T`で指定 |
| 学習上の要点 | `_fast_solves`とPyTorch optimizer | `use_iterative_methods()` |
| 固有のraw API | `raw_train_X`, `raw_train_Y` | 左に加えて`raw_train_T` |
| Mixed版 | `MixedHigherOrderGP` | `MixedLatentKroneckerGP` |

いずれも`raw_train_Yvar`は`None`であり、raw値は変換・masking前の入力snapshotである。
raw値と内部の変換済み・masking済みデータを区別する。

少数の独立出力なら`ModelListGP`、関連タスクなら`MultiTaskGP`や
`KroneckerMultiTaskGP`も検討する。Kronecker構造の利用だけでは、
タスク共分散と物理出力座標のモデルを同一視できない。
出力数が多くても軸構造がなければ、出力削減・潜在モデルが適することがある。
Kronecker計算の効率は分離可能共分散という仮定と引き換えである。

## BOへ接続するときの目的と検証範囲

代理モデルは応答全体の分布を返す。BO側では、特定位置、ピーク、積分値、
目標スペクトルからの距離など、何を最大化するかを別に定義する。
非線形functionalはposterior sampleへ適用し、不確実性を伝播する。
平均値へfunctionalを適用した結果が期待utilityと一致するとは限らない。
Objective / PosteriorTransformと獲得関数がtensorの軸を扱えるかも確認する。

[HOGPテスト](../../tests/models/structured/test_higher_order_gp.py)と
[LatentKroneckerテスト](../../tests/models/structured/test_latent_kronecker_gp.py)は、
上流とのコンストラクタ・posteriorの互換性、raw保持、dtype変換などを確認する。
個々の数値一致と、学習の収束・全structured objectiveへの互換性は別の検証である。
[Structured Output Notebook](../../examples/notebooks/11_structured_output_gp.ipynb)に
学習・posterior比較・新しい`test_T`での予測例がある。

## 参考文献

1. Zhe, S., Xing, W., and Kirby, R. M. (2019).
   Scalable high-order Gaussian process regression. *Proceedings of AISTATS*.
2. BoTorch `HigherOrderGP` and `LatentKroneckerGP` documentation.
3. [Gaussian Process](../theory/02_gaussian_process.md)、
   [Kernel](../theory/03_kernel.md)、[Multi-task / Multi-output](../theory/07_multitask_multioutput.md)。

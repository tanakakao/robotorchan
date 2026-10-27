# Structured Output / Higher-Order GP：理論と実装

## 解決する問題と数理の入口

[理論の正本](../../../theory/11_structured_output.md)は数学的仮定・導出を担当する。本章は入力データ、アルゴリズム、公開API、上流委譲、数値検証のつながりを日本語で説明する。

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

コードは確認した公開APIを示す最小例であり、今回のセッションで実行して成功したとの主張ではない。tensorのshape、dtype/device、入力変数の意味と、posteriorの出力構造を意識して利用する。

## 検証・未検証事項

上の例はモデル構築までの使用形態であり、tensor-output posteriorの完全なshape/数値検証や学習成功を宣言するものではない。tensor出力を単純に潰して単一出力posteriorとして扱わない。関連：[Structured Outputモデル](../../../models/structured_output.md)。

[旧Model Mapping](../models.md)および[検証記録](../verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../../../models/README.md)を参照する。

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

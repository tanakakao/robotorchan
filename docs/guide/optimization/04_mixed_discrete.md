# Mixed / Discrete：獲得関数最適化の実装ガイド

## 1. この章の問い

整数は順序・距離があるがunordered category IDに自然な距離はない。fixed features列挙とtype-aware変異を選び分ける。

数理的な正本は[理論の対応章](../../theory/optimization/04_mixed_discrete.md)とする。本章は、そこで定義される問題の**候補の座標系、実装経路、契約、検証上の限界**を学ぶための日本語案内である。

## 2. 理論から実装まで

```text
surrogate.posterior(X)
      ↓
acquisition α(X) ← objective / constraints
      ↓
optimizer: bounds + fixed features + candidate feasibility
      ↓
candidate X*（joint q × d）
      ↓
実際のblack-box評価 → 次の観測を追加
```

BoTorch `optimize_acqf_mixed`とrobotorchanのmixed/backends、categorical assignmentや整数repairを切り分ける。

| 役割 | 実装の追跡先 | 検証すべき点 |
| --- | --- | --- |
| 数学的目的 | `docs/theory/optimization/04_mixed_discrete.md` | αを最大化する変数・制約・近似 |
| 公開API / dispatcher | `src/robotorchan/optim/` | 対応するoptimizer名、能力宣言、fallback |
| 具体的な探索backend | `src/robotorchan/optim/`とBoTorch optimizer | 実際の探索アルゴリズム、依存ライブラリ |
| 数値実行証拠 | [旧Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md) | test assertions、実装境界、未対応条件 |

実装方法は**Implemented / Extended / Delegated**を分けて読む。BoTorchへ処理を委譲する名前と、robotorchanが実際に更新則を実装する名前を同等の独自アルゴリズムとして数えない。実際のbackend、candidate制約とposteriorの相互依存は旧Mappingを参照する。

## 3. 適用条件と失敗の解釈

categorical codeを連続補間して丸める方法は人工的な順序を持ち込む。q-batchでのカテゴリ割当も個別に確認する。

最適化器が候補を返すことと、その候補が厳密な大域最適解であることは違う。候補座標の型、bounds、fixed features、constraint tolerance、q軸、one-shot追加変数、random seed等を適用先に合わせる。モデルの学習後posteriorが有限でも、candidate探索の収束は別途確認する。

## 4. 検証と使い方の入口

最低限の確認順序は、**(1) 問題の型→(2) 数理契約→(3) optimizer dispatch→(4) backend→(5) 制約と出力shape→(6) 再現実験**である。CI成功だけで全backend・全constraint・全qに対して正しいとは主張しない。

[Optimization利用ガイド](../../optimization/README.md)、[理論Optimization索引](../../theory/optimization/README.md)、[旧Optimization Mapping](../../../.github/project-records/theory-guide/mapping/optimization.md)、[Verification](../../../.github/project-records/theory-guide/mapping/verification.md)を参照。原文は章別照合完了まで削除しない。

## 5. Native mixed BoTorchとmixed one-shotの数学的差異

旧Optimization Mappingの固定区間246–257と src/robotorchan/optim/backends/botorch.py を照合すると、native mixed経路は optimize_acqf_mixed_botorch から BoTorch optimize_acqf_mixed へ委譲する。有限なカテゴリ割当て集合 C に対して、各割当て c で連続変数の acquisition を最大化する問題と考えられる。

$$
\max_{c\in C}\;\max_{x_{\mathrm{continuous}}}
\alpha(x_{\mathrm{continuous}},c).
$$

固定するカテゴリ値は fixed_features_list として渡される。これは11種類の named optimizer を選択する optimize_acqf(optimizer="botorch") とは異なり、後者は mixed MixedVariableSpace を明示的に拒否する。数式は探索方針の概念化であり、BoTorch内部の全枝刈りやq-batch処理の完全な再記述ではない。

| 比較対象 | 実装入口 | 主要な制約 |
| --- | --- | --- |
| native mixed | backends/botorch.py::optimize_acqf_mixed_botorch | fixed_features_list を受け、inter-point nonlinear constraintsを拒否 |
| mixed GA | dispatch.pyの optimizer="ga" と MixedVariableSpace | is_mixedならmixed_genetic_algorithm.pyに分岐 |
| mixed one-shot | one_shot/mixed.py::optimize_mixed_one_shot_acqf | fantasy補助行を含む拡張batchに行別カテゴリ割当てが必要。q=1のみ |

native mixedではnonlinear candidate constraintsを使用すると、feasibleな batch_initial_conditions または ic_generator が必要となる。inter-point nonlinear制約はValueErrorで拒否され、q点間の制約を点単位の独立制約へ無断変換しない。候補の数値精度や全制約可行性は、委譲したという事実だけから保証できない。

通常のカテゴリIDを連続座標として補間することと、カテゴリごとの固定割当てを明示することは異なる。特にone-shotのfantasy行には別々のカテゴリ値が入り得るため、[Batch/one-shot](08_batch_one_shot.md)の拡張batch契約を参照する。

**証拠水準：** 上記の関係はソースの静的照合であり、独立した探索性能・数学的最適性の数値再実行ではない。

## 6. 固定カテゴリ割当てによる候補探索

次の例は、2列の候補で第2列を`0`または`1`に固定して
連続の第1列を最適化する。カテゴリを連続補間して丸めるのではなく、
`fixed_features_list`の各割当てをBoTorchへ渡す。

**注意：** この小例の`SingleTaskGP`は数値列を通常の連続入力として学習する。
したがってカテゴリkernelの学習を検証した例ではなく、
mixed optimizerの固定割当て契約だけを示す。
実際の非順序カテゴリ予測には`MixedSingleTaskGP`などを検討する。

```python
import torch
from botorch.acquisition.analytic import PosteriorMean
from robotorchan.models import SingleTaskGP
from robotorchan.optim.backends.botorch import optimize_acqf_mixed_botorch

base = torch.linspace(0, 1, 6, dtype=torch.double)
low = torch.stack((base, torch.zeros_like(base)), dim=-1)
high = torch.stack((base, torch.ones_like(base)), dim=-1)
train_X = torch.cat((low, high), dim=0)
train_Y = torch.sin(4 * train_X[:, :1]) + 0.1 * train_X[:, 1:]
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
model.eval()
acqf = PosteriorMean(model=model)
bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
candidate, value = optimize_acqf_mixed_botorch(
    acq_function=acqf,
    bounds=bounds,
    q=1,
    num_restarts=2,
    fixed_features_list=[{1: 0.0}, {1: 1.0}],
    raw_samples=16,
)
assert candidate.shape == (1, 2)
assert candidate[0, 1].item() in (0.0, 1.0)
assert torch.isfinite(value).all()
```

ここでは連続列のboundsと列1の離散的割当てを直接確認する。
整数変数のrepair、複数カテゴリ、q-batch、制約付きmixed探索の
全組合せを検証したことにはならない。

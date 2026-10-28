# Differential Evolution backend

SciPy `differential_evolution` を使う微分不要の最適化器です。BoTorch の `AcquisitionFunction`、Tensor の bounds、q-batch の概念を維持します。

joint 候補 `q x d` を `q*d` の決定ベクトルに平坦化し、各評価で獲得関数の device・dtype に復元して獲得値の負を最小化します。

進化計算の制御は CPU 側の SciPy が行います。獲得関数自体はモデルの Tensor device 上で評価できますが、GPU 対応や集団のベクトル化評価を保証するものではありません。

候補制約は共通の違反量評価から SciPy `NonlinearConstraint` へ変換します。既定で polishing を無効にし、局所最適化へ暗黙に切り替えません。

```python
from robotorchan.optim.backends import optimize_acqf_de

candidates, value = optimize_acqf_de(
    acq_function=acq,
    bounds=bounds,
    q=3,
    seed=123,
    options={"maxiter": 200, "popsize": 15},
)
```



## 整数変数

整数座標も DE の決定ベクトルに含めます。獲得関数・制約の評価前と返却前に、最寄りの合法整数へ丸めて bounds に収めます。

`integer_dims` の直接指定と、連続・整数を宣言した `MixedVariableSpace` に対応します。両者の同時指定は、探索空間の正本を一つに保つため拒否します。

## カテゴリ変数と混合空間

カテゴリ座標は評価前に明示的な集合の最寄りの合法値へ写し、補間されたカテゴリ値を獲得関数や制約へ渡しません。整数座標は独立に丸め・切り詰めます。連続・整数・カテゴリを組み合わせた領域にもこの規則を適用します。

カテゴリ符号は数値提案を決定的な割当領域へ分けるために使います。カテゴリ間の距離に意味があることや、SciPy の差分変異自体がカテゴリ演算になることを主張しません。

返り値にも同じ修復を適用します。`CandidateConstraints` は修復後の元座標を受け取り、獲得関数と同じ候補を評価します。

## 制約処理

固定の罰則係数で獲得値と違反量を合算せず、共通の違反量評価から SciPy の非線形実行可能性制約を構成します。集団は実行可能性を目的値より優先し、獲得値のスケールに依存する罰則を使いません。

領域の修復と fixed features の適用は獲得値・制約の両評価より前に行います。公開契約は `CandidateConstraints` のままで、DE 固有の制約 DSL は追加しません。

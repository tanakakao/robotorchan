# Candidate constraint support matrix

この表は `src/robotorchan/optim/capabilities.py` と公開dispatcher、
各backendの制約処理に基づく**API対応範囲**です。すべての組合せの実行成功を
保証するものではありません。制約は入力候補Xに対するものであり、出力Yの制約とは別です。

## 公開 `optimize_acqf` の対応

| optimizer | 線形不等式 | 線形等式 | 非線形不等式 | 非線形等式 | 整数・カテゴリ | q>1 |
| --- | --- | --- | --- | --- | --- | --- |
| botorch | ○ | ○ | ○ | ○* | ×（専用Mixed APIあり） | ○ |
| torch_adam / adamw / sgd | △ | △ | △ | △* | × | ○ |
| random / sobol | ○ | ○ | ○ | ○* | ○ | ○ |
| de | ○ | ○ | ○ | ○* | ○ | ○ |
| cmaes | ○ | ○ | ○ | ○* | ○（Mixed経路） | ○ |
| ga | ○ | ○ | ○ | ○* | ○（Mixed GAへ分岐） | ○ |
| pso | ○ | ○ | ○ | ○* | ○ | ○ |
| hybrid | ○ | ○ | ○ | ○* | ○（制限あり） | ○（制限あり） |

**凡例:** ○ = APIが制約型を受け付ける、△ = ペナルティ探索・最終可行性検査
（成功保証ではない）、× = この公開経路では非対応。○は**任意の制約を満たす解が必ず得られる**
という意味ではありません。

*非線形等式はBoTorchのnative等式APIではありません。
`nonlinear_equality_constraints` の各 `h(X)=0` を
`h(X)+epsilon >= 0` および `epsilon-h(X) >= 0` に変換します。
`nonlinear_equality_tolerance` はこの `epsilon` です。
厳密なゼロ残差は保証しません。

## q-batch・Mixedの制限

- 候補間制約（inter-point）はjoint q-batchが必要です。逐次最適化では扱えません。
- BoTorch Mixedは専用APIでfixed featuresを列挙します。
  inter-point nonlinear制約や、一部のq>1 inter-point linear制約は非対応です。
- Mixedの線形制約でカテゴリ座標を直接参照することはできません。
  整数座標の扱いはbackend固有の制限に従います。
- Mixed Hybridはq>1の候補間線形制約をローカル段階で維持できないため拒否します。
- NSGA-IIは上表のscalar acquisition dispatcherとは別APIです。
  Mixed NSGA-IIでは整数座標の線形不等式に対応しますが、
  カテゴリ座標を含む線形制約や整数・カテゴリの線形等式には制限があります。
- 非線形等式を含む全組合せについて、テスト済みであることを意味しません。
  対象backendの実装・既存テストを確認してください。

## 制約処理方式

| 経路 | 処理 |
| --- | --- |
| BoTorch | native constrained optimization（非線形等式は不等式へ変換） |
| Torch | 微分可能なペナルティ、既定strictモードで最終候補を検査 |
| Random / Sobol | サンプルを可行性でフィルタ。可行サンプルがなければエラー |
| DE / CMA-ES / GA / PSO | feasibility-first系。可行解発見を保証しない |
| Hybrid | global探索後のlocal refinementを検査し、必要なら可行なglobal候補を維持 |

## 実行例

- [Acquisition Optimization Notebook](../../examples/notebooks/31_acquisition_optimization.ipynb):
  線形・候補間制約とjoint q-batch
- [Nonlinear Candidate Constraints Notebook](../../examples/notebooks/38_nonlinear_candidate_constraints.ipynb):
  非線形不等式、非線形等式（許容誤差）、候補間非線形制約
- [Candidate constraint Guide](../guide/optimization/05_constraints.md):
  数学的契約、backendへの委譲、実装上の制限

この表は実装契約のスナップショットです。PyPI公開前には、各backendの
実行可能な代表ケースとCI結果を別途確認してください。

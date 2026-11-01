# Gradient-based：獲得関数最適化の実装ガイド

## この章の問い

必要なのはcandidate Xに対する∇α(X)であり、GPパラメータ学習の勾配とは違う。reparameterized MCサンプルで候補勾配を評価できる。

数理的な正本は[理論の対応章](../../theory/optimization/02_gradient_based.md)とする。本章は、そこで定義される問題の**候補の座標系、内部処理、
委譲と設計上の制限**を学ぶための日本語案内である。

## 適用条件と失敗の解釈

入力gradientが存在することだけではoptimizerや非線形制約への適合性は証明されない。

この章のBoTorch backendは複数の初期点から局所探索する。`num_restarts`と`raw_samples`は学習データ数ではなく探索初期化の設定である。
返却候補が有限でbounds内にあっても、非凸獲得関数の大域最大値であることまでは保証しない。
非線形制約の可行性は[Candidate constraints](05_constraints.md)で別に検証する。

## 勾配計算と更新則を担当する層

[BoTorch backend](../../../src/robotorchan/optim/backends/botorch.py)の
`optimize_acqf_botorch()`は制約と初期条件を整え、BoTorchの`optimize_acqf()`へ委譲します。
上流の`gen_candidates_scipy()`は候補Tensorで獲得関数と勾配を評価し、負の獲得値をSciPy側の最小化問題として扱います。
候補に関する自動微分とGPハイパーパラメータの学習は別の処理です。

[torch backend](../../../src/robotorchan/optim/backends/torch.py)の
`optimize_acqf_torch()`は、`gen_candidates_torch()`へ渡すPyTorch optimizerを選びます。
restartの初期化と最良候補の選択は上流BoTorchへ残し、数値候補生成器だけを差し替えます。Adam・AdamW・SGDの更新則はPyTorchが担当し、
robotorchanの独自勾配法ではありません。この経路はcandidate constraintsを明示的に拒否します。

BoTorch側の候補生成は[generation/gen.py][gen]、多初期値の制御は[optim/optimize.py][optimize]にあります。局所探索と多初期値探索は、
非凸問題の大域最適性を一般に保証する仕組みではありません。

[gen]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/generation/gen.py
[optimize]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/optim/optimize.py

使用手順と結果の可視化は[対応するNotebook](../../../examples/notebooks/31_acquisition_optimization.ipynb)を参照してください。

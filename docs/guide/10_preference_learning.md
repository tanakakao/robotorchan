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

比較は`(勝者index, 敗者index)`の契約で扱う。比較グラフの連結性、正則化、Laplace approximationと潜在utilityの尺度を区別する。静的API検査を「全ての比較集合で学習成功」と読み替えない。既存テストとNotebookは後半の節から参照できる。

[旧Model Mapping](../../.github/project-records/theory-guide/mapping/models.md)および[検証記録](../../.github/project-records/theory-guide/mapping/verification.md)は移行元の正本として保持する。API存在、静的assertion、CI成功、独立数値検証は区別する。使用例と詳しいモデル選択は[モデルガイド](../models/README.md)を参照する。

## Pairwise likelihoodとLaplace MLLの実装上の違い

候補xiとxjの潜在utility u(x)について、比較データがiをjより好むと表す場合は、確率的にP(i≻j|u_i,u_j)をlikelihoodで定義する。比較の行に観測されるのは連続値y_iではなく、候補のindexの組である。

src/robotorchan/models/preference/pairwise.pyのPairwiseGPはBoTorch PairwiseGPを継承し、datapoints・comparisons・pairwise likelihood・入力変換を親へ渡す。Laplace approximation、重複点のconsolidation、utility posteriorの演算はBoTorchの担当となり、robotorchanはconstructorのraw_datapoints/raw_comparisonsをdetach/cloneしたsnapshotとして保持する。

make_mll()はPairwiseLaplaceMarginalLogLikelihood(self.likelihood,self)を返す。ExactGPModelMixinのExactMarginalLogLikelihoodでも、VariationalELBOでもない。例えばcomparisons=[[1,0],[1,2]]は候補1が0と2に優越する観測を表すが、utilityの絶対尺度・ランキングの全順序・データの比較整合性を自動保証しない。

priorとlikelihoodの指定、comparison indexの有効範囲、モデルの数値条件、Laplace近似の収束は個別に検査すべき事項である。現行の`PairwiseGP.fit()`は`fit_gpytorch_mll(self.make_mll())`へ委譲する。全モデルに同じ`fit()`シグネチャがあるわけではない。

**検証境界：** ソースのconstructor・make_mll・raw保持を照合。比較グラフに対するposteriorの数学的整合、収束、選好予測性能はこの説明だけでは独立数値テストの完了を保証しない。

## カーネル・尤度・utilityの尺度

BoTorchの既定構成はscaled RBF kernelとprobit likelihoodを使う。
Bradley–Terryモデルのlogistic linkと、Gaussian CDFによるprobit linkを区別する。
比較尤度のノイズ尺度と潜在utilityの尺度は、通常の数値回帰と同じ形では識別できない。
BoTorchの既定likelihoodでは独立な観測ノイズ分散を推定する代わりに、
`ScaleKernel`のoutput scaleも含めたutilityの尺度を扱う。
通常の回帰GPの`observation_noise=True`を、同じ意味の比較ノイズ指定と解釈しない。

学習では、比較とGP priorの下で潜在utilityのMAPを求める内部処理と、
Laplace近似MLLを通じたカーネルのハイパーパラメータ最適化を区別する。
utilityの絶対値には任意性があるため、ランキングや比較確率を目的に合う指標で評価する。
MAPとLaplace近似を必要とする分、通常のExact GP回帰より学習が重くなる場合がある。

## 学習・予測とrawデータ

コンストラクタ例に続けて、モデル固有の学習経路を使用できる。

```python
model.fit(optimizer_kwargs={"options": {"maxiter": 50}})
posterior = model.posterior(points)
assert torch.isfinite(posterior.mean).all()
assert torch.isfinite(posterior.variance).all()
```

`make_mll()`を用いてBoTorchの学習関数を直接呼ぶ方法も使える。
`raw_datapoints`と`raw_comparisons`は、重複点の統合や入力変換前のコンストラクタ入力である。
`raw_train_Y`は提供しない。新しい比較を`condition_on_observations()`へ渡す場合の`Y`も
実数値の回帰ターゲットではなく、BoTorchの比較index契約に従う。

[Pairwise wrapperテスト](../../tests/models/preference/test_pairwise_gp.py)は、
コンストラクタの互換性、raw snapshot、重複統合前のデータ保持、MLLの型、
上流posteriorとの数値一致を確認する。
[学習API](../development/training_api.md)では`fit()`の系統別契約を説明する。
この検証は全比較グラフでの近似収束や選好の較正を保証しない。

## 次に比較する候補とBOPE

`PairwiseGP`は比較データからutility posteriorを作る代理モデルである。
次の比較対象は、EUBOなどのpreference acquisitionと対応する候補最適化で選ぶ。
BoTorchのpreference BOには`q=2`で比較する2候補を選ぶ構成がある。

```text
候補の比較を観測
  → PairwiseGPを学習
  → utility posterior
  → EUBO等で次のquery pairを選ぶ
  → 新しい比較をcomparisonsへ追加
```

BOPEでは、測定可能な複数outcomeの代理モデルと、outcomeに対する人の選好モデルを分ける。
例えば材料特性を数値測定し、総合的な好みを比較で取得する場合に検討する。
単一`PairwiseGP`を二値分類器やGaussian回帰器として扱わない。

| 得られる観測・仮定 | 検討する構成 |
| --- | --- |
| 信頼できる実数スコア | `SingleTaskGP`などの回帰モデル |
| 候補間の優劣 | `PairwiseGP`とpreference acquisition |
| 複数outcomeの数値と総合的な選好 | BOPE型のoutcome / utility分離 |
| 絶対的な二値ラベル | [分類モデル](24_classification.md) |
| 順序尺度のスコア | 観測形式に合うordinal model |
| 評価者の個人差 | 評価者構造を表すpreference modelを検討 |
| 強い循環選好 | 単一のscalar utility仮定を再検討 |

最後の3行は問題のモデル化を検討するための分類であり、
robotorchanが専用のordinal・評価者別モデルを提供するという宣言ではない。
比較が不整合な場合や個人差が大きい場合も、単一utilityへ無理に集約しない。

## 実行例と参考文献

APIの詳細は[Preferenceモデルガイド](../models/preference_learning.md)、
学習から候補生成までは[Preference Notebook](../../examples/notebooks/07_pairwise_gp.ipynb)を参照する。

1. Chu, W., and Ghahramani, Z. (2005). Preference learning with Gaussian processes.
   *Proceedings of ICML*, 137–144.
2. Lin, Z. J. et al. (2022). Preference exploration for efficient Bayesian optimization
   with multiple outcomes. *Proceedings of AISTATS*.
3. Astudillo, R. et al. (2023). qEUBO: A decision-theoretic acquisition function
   for preferential Bayesian optimization. *Proceedings of AISTATS*.
4. BoTorch `PairwiseGP`, preference BO, and BOPE documentation.
5. [Gaussian Process](../theory/02_gaussian_process.md)と
   [Acquisition Function](../theory/04_acquisition_function.md)。

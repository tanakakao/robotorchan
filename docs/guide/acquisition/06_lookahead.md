# Lookahead / Knowledge Gradient：理論から実装へ

## 1. この章で解決する問い

KGは観測後の最終意思決定価値の期待的改善を測る。EIは現在候補のimprovement、MESは未知量の情報利得を扱う。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/06_lookahead.md)が正本であり、この章は**数式の計算対象 → 実装の所有権 → 公開API → 探索上の制約**をつなぐ日本語実装ガイドである。

## 2. posteriorからcandidateへの処理の流れ

```text
train observations → surrogate.fit / posterior(X)
                                     ↓
                          objective / utility
                                     ↓
                  acquisition function α(X)
                                     ↓
                    candidate optimizer
                                     ↓
             selected X → costly evaluation
```

候補点で観測する未知の応答値と、観測前に計算する獲得関数値は異なる。獲得関数が最大化するutilityを、posterior meanの予測結果と読み替えない。analytic計算、MC sampling、fantasization、classification probability等、用いる数値表現は手法とモデルによって異なる。

## 3. 実装の対応と委譲

`qKnowledgeGradient`・`qMultiFidelityKnowledgeGradient`はBoTorch-native。fantasizeとone-shot augmented candidate optimizationが必要で、通常のmixed optimizer経路と異なる。

参照する代表的な公開APIは ``botorch.acquisition.knowledge_gradient.qKnowledgeGradient``。ここでは**名前・責務の参照**であり、すべてのモデルやoptimizerとの組合せを実行検証済みとするものではない。

| 層 | 何を担当するか | 検証上の要点 |
| --- | --- | --- |
| surrogate | posterior、必要に応じてclass probability | posterior type、出力軸、学習方式 |
| acquisition | utilityの期待値・近似または探索score | 定義した数式、方向、q、sample形状 |
| semantic bridge | objective/constraintの意味論変換 | objectiveとclassification feasibilityの分離 |
| optimizer | 入力境界、fixed features、candidate constraintsを満たす候補検索 | one-shot/gradient/mixedの対応、最適化失敗の検出 |

## 4. 採用時の確認と注意事項

`one_shot=True`のmetadataと`MIXED_ONE_SHOT`経路の対応、fantasy points、cost utilityを確認。

モデル側のposteriorが正しい形を返すことと、獲得関数の数学的同値性を示すことは別である。またacquisition functionの単体テストと、候補探索が可行な点を返すことは別々に検証する。

## 5. 数学的検証と実装の追跡

[既存Acquisition Mapping](../../../.github/project-records/theory-guide/mapping/acquisition.md)はsource inspectedの実装契約、entry・テスト証拠を保持する。[Verification Mapping](../../../.github/project-records/theory-guide/mapping/verification.md)は独立数値参照や既存のassertionの追跡先とする。両者を消去・置き換えず、原文区間のmigration ledgerをverifiedとする際に照合する。CI成功だけで全数値契約が検証されたことにはならない。

[獲得関数の基礎ガイド](../04_acquisition_function.md)、[獲得関数利用ドキュメント](../../optimization/README.md)、[理論Acquisition索引](../../theory/acquisition/README.md)にも戻れる。個別の数式、適用例や比較条件は対応する理論章と使用ガイドで読む。


## 6. KGの数式とone-shot拡張batchの対応

旧Acquisition Mappingの固定区間346–362にあるKGは、候補観測後に最良の決定を**選び直せる価値**を測る。単一候補の場合の概念式は

$$
\alpha_{\mathrm{KG}}(x)=
\mathbb E_{Y_x\mid\mathcal D}
\left[\max_{z\in\mathcal X}\mathbb E[f(z)\mid\mathcal D,(x,Y_x)]\right]
-\max_{z\in\mathcal X}\mathbb E[f(z)\mid\mathcal D].
$$

右辺の内側の最大化はfantasy observation後の意思決定であり、現在候補のimprovementであるEI、最適値のentropyであるMESとは異なる。BoTorchの実際のqKGは複数候補・fantasy sample・terminal valueの設定に対応するので、上式はその全数値実装を定義したものではない。

| 経路 | APIとlocal capability | 利用上の制約 |
| --- | --- | --- |
| qKG | BoTorch qKnowledgeGradient。registryでone_shot=True、requires_fantasize=True、supports_multi_output=False | augmented q-batchに実験候補とfantasy decision variablesが共存する |
| Multi-fidelity KG | BoTorch qMultiFidelityKnowledgeGradient。registryでrequires_multi_fidelity=True | fidelity-aware model、cost/value設計が別途必要。metadataはcost-aware policyを自動生成しない |
| Multi-step | BoTorch qMultiStepLookaheadを直接利用 | local registryに名前付きentryなし。複数future stageのfantasy treeが増大する |

具体的には、BoTorchのone-shot acquisitionは実験するq行だけでなく補助decision variableを最適化変数として受け取る。batch_initial_conditionsを指定するなら acqf.get_augmented_q_batch_size(q) 行に合わせ、optimize_acqfの結果はextract_candidatesによって実験対象のq行のみを取り出す。fantasy補助点を実験候補として解釈してはいけない。

~~~python
import torch
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from robotorchan.models import SingleTaskGP

train_x = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.double)
train_y = torch.sin(train_x * 5.0)
model = SingleTaskGP(train_x, train_y)
acqf = qKnowledgeGradient(model=model, num_fantasies=16)
augmented_q = acqf.get_augmented_q_batch_size(1)
assert augmented_q == 17
augmented_candidates = torch.linspace(0.1, 0.9, augmented_q, dtype=torch.double).reshape(
    1, augmented_q, 1
)
experimental_candidates = acqf.extract_candidates(augmented_candidates)
assert experimental_candidates.shape == (1, 1, 1)
assert torch.equal(experimental_candidates, augmented_candidates[..., :1, :])
~~~

BoTorchのoptimize_acqfにはqKG専用initializerがあり、robotorchanはそれを置き換えない。qMultiStepLookaheadには通常のinitializerだけではfantasy tree全体の初期条件を作れないため、docs/optimization/lookahead.mdの通りgen_augmented_one_shot_initial_conditionsをic_generatorとして使うか、拡張batch用のbatch_initial_conditionsを明示する。stage数・samplerのfantasy分岐数とともに最適化変数も増える。

src/robotorchan/acquisition/optimizer_compatibility.pyのone-shot compatibility判定では通常のMIXEDを拒否し、MIXED_ONE_SHOTをone-shotに限定する。ただしこれは静的契約であって、実行したoptimizerの可行性・解の品質・cost-aware最適化を検証しない。

**検証の限界：** src/robotorchan/acquisition/registry.py、docs/optimization/lookahead.md、旧Mappingを静的照合した。今回独立の数値実験は行っていない。移行区間346–362の判定はpartial / review requiredのままとする。

## 実装補足：理論本文から分離した対応情報

### 6.17 BoTorch / robotorchan との対応

BoTorch は `qKnowledgeGradient` と `qMultiStepLookahead` を提供し、fantasy model を使った lookahead
optimization をサポートします。

robotorchan はこれらの標準手法を再実装せず、BoTorch native path を基本とします。

利用上の位置付けは [Lookahead optimization guide](../../optimization/lookahead.md)、現在の統合状況は [Acquisition
integration status](../../../docs-internal/contracts/optimization/acquisition-integration.md) を参照してください。

## 理論章から分離した実装上の条件

#### 6.7 Inner optimization：実装上の補足

BoTorch では fantasy points / fantasy samples と acquisition optimization を組み合わせてこの構造を扱います。

#### 6.11 One-shot formulationとauxiliary decision rows：実装上の補足

nested inner optimizationをそのまま繰り返す代わりに、BoTorchのone-shot lookahead acquisitionでは
actual candidateとfuture / fantasy decision variablesを一つのaugmented optimization problemへ
組み込む場合があります。

#### 6.12 qMultiStepLookahead：実装上の補足

BoTorch の `qMultiStepLookahead` は、複数段階の fantasy decisions を含む lookahead acquisition
を構成するための一般的な枠組みです。

#### 6.19 まとめ：実装上の補足

- Frazier, P. I., Powell, W. B., and Dayanik, S. (2008), *A Knowledge-Gradient Policy for
Sequential Information Collection*. SIAM Journal on Control and Optimization.
- Jiang, S. et al. (2020), *Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step
Trees*. NeurIPS.
- BoTorch documentation, *Acquisition Functions*, for current Knowledge Gradient and multi-step
lookahead APIs.

## 数理上の概念と具体的な識別子の対応

以下は理論で一般的な概念名に分けた箇所の、具体的なモデル・変数・処理の対応です。数学的な導出は対応する Theory 章を参照してください。

- KGはcandidate valueではなく観測後のterminal decision valueの改善を評価する
- MESのinformation gainとKGのvalue of informationは同じ基準ではない
- fantasy outcome / modelはreal `X_pending` と別物
- qKGはpointwise KG上位q点ではなくjoint future observationsを扱う
- batch qとlookahead depthは別の軸
- multi-stepでは後段decisionが前段observationへadaptiveに依存する
- one-shot augmented rowsのすべてが実評価candidateではない
- qKGの現在のregistry contractはposterior samples、single-output、fantasize、one-shotを要求する
- mixed one-shotは通常の `optimize_acqf_mixed` ではなく専用optimizerを使う
- 現在のmixed one-shot candidate generationはq=1のbounded exact-enumeration contractである

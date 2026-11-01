# Lookahead / Knowledge Gradient：理論から実装へ

## この章で解決する問い

KGは観測後の最終意思決定価値の期待的改善を測る。EIは現在候補のimprovement、MESは未知量の情報利得を扱う。

数学的な定義・仮定・導出は[理論の対応章](../../theory/acquisition/06_lookahead.md)が正本であり、
この章は**数式の計算対象 → クラス構造 → 内部処理 →近似と制限**をつなぐ日本語実装ガイドである。

## 実装の対応と委譲

`qKnowledgeGradient`・`qMultiFidelityKnowledgeGradient`はBoTorch-native。
fantasizeとone-shot augmented candidate optimizationが必要で、通常のmixed optimizer経路と異なる。

## 採用時の確認と注意事項

`one_shot=True`のmetadataと`MIXED_ONE_SHOT`経路の対応、fantasy points、cost utilityを確認。

## KGの数式とone-shot拡張batchの対応

$$
\alpha_{\mathrm{KG}}(x)=
\mathbb E_{Y_x\mid\mathcal D}
\left[\max_{z\in\mathcal X}\mathbb E[f(z)\mid\mathcal D,(x,Y_x)]\right]
-\max_{z\in\mathcal X}\mathbb E[f(z)\mid\mathcal D].
$$

右辺の内側の最大化はfantasy observation後の意思決定であり、現在候補のimprovementであるEI、最適値のentropyであるMESとは異なる。
BoTorchの実際のqKGは複数候補・fantasy sample・terminal valueの設定に対応するので、上式はその全数値実装を定義したものではない。

- **qKG**
  APIとlocal capability：BoTorch qKnowledgeGradient。registryでone_shot=True、requires_fantasize=True、
  supports_multi_output=False
  利用上の制約：augmented q-batchに実験候補とfantasy decision variablesが共存する

- **Multi-fidelity KG**
  APIとlocal capability：BoTorch qMultiFidelityKnowledgeGradient。registryでrequires_multi_fidelity=True
  利用上の制約：fidelity-aware model、cost/value設計が別途必要。metadataはcost-aware policyを自動生成しない

- **Multi-step**
  APIとlocal capability：BoTorch qMultiStepLookaheadを直接利用
  利用上の制約：local registryに名前付きentryなし。複数future stageのfantasy treeが増大する

具体的には、BoTorchのone-shot acquisitionは実験するq行だけでなく補助decision variableを最適化変数として受け取る。
batch_initial_conditionsを指定するなら acqf.get_augmented_q_batch_size(q) 行に合わせ、
optimize_acqfの結果はextract_candidatesによって実験対象のq行のみを取り出す。fantasy補助点を実験候補として解釈してはいけない。

BoTorchのoptimize_acqfにはqKG専用initializerがあり、robotorchanはそれを置き換えない。
qMultiStepLookaheadには通常のinitializerだけではfantasy tree全体の初期条件を作れないため、
docs/optimization/lookahead.mdの通りgen_augmented_one_shot_initial_conditionsをic_generatorとして使うか、
拡張batch用のbatch_initial_conditionsを明示する。stage数・samplerのfantasy分岐数とともに最適化変数も増える。

src/robotorchan/acquisition/optimizer_compatibility.pyのone-shot compatibility判定では通常のMIXEDを拒否し、
MIXED_ONE_SHOTをone-shotに限定する。ただしこれは静的契約であって、実行したoptimizerの可行性・解の品質・cost-aware最適化を検証しない。

### BoTorch / robotorchan との対応

BoTorch は `qKnowledgeGradient` と `qMultiStepLookahead` を提供し、
fantasy model を使った lookahead optimization をサポートします。

robotorchan はこれらの標準手法を再実装せず、BoTorch native path を基本とします。

利用上の位置付けは [Lookahead optimization guide](../../optimization/lookahead.md)、
現在の統合状況は
[統合契約](../../../docs-internal/contracts/optimization/acquisition-integration.md)
を参照してください。

## 補足となる実装上の設計

#### Inner optimization：実装上の補足

BoTorch では fantasy points / fantasy samples と acquisition optimization を組み合わせてこの構造を扱います。

#### One-shot formulationとauxiliary decision rows：実装上の補足

nested inner optimizationをそのまま繰り返す代わりに、
BoTorchのone-shot lookahead acquisitionではactual candidateとfuture / fantasy decision
variablesを一つのaugmented optimization problemへ組み込む場合があります。

#### qMultiStepLookahead：実装上の補足

BoTorch の `qMultiStepLookahead` は、複数段階の fantasy decisions を含む lookahead
acquisitionを構成するための一般的な枠組みです。

#### まとめ：実装上の補足

- Frazier, P. I., Powell, W. B., and Dayanik, S. (2008), *A Knowledge-Gradient Policy for
Sequential Information Collection*. SIAM Journal on Control and Optimization.
- Jiang, S. et al. (2020), *Efficient Nonmyopic Bayesian Optimization via One-Shot Multi-Step
Trees*. NeurIPS.
- BoTorch documentation, *Acquisition Functions*, for current Knowledge Gradient and multi-step
lookahead APIs.

## 数理モデルとクラス構造

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

## fantasy treeと最終価値の計算

BoTorchの`qKnowledgeGradient`は`MCAcquisitionFunction`と`OneShotAcquisitionFunction`を継承します。
`forward()`は拡張入力を実候補と補助decisionへ分け、モデルの`fantasize()`で観測後モデルを構成して、その上の終端価値を評価します。
`get_augmented_q_batch_size()`と`extract_candidates()`は、この内部変数と実験候補の境界を明示するメソッドです。

`qMultiStepLookahead`では`get_multi_step_tree_input_representation()`が拡張入力をstageごとのtree表現へ戻し、
後段decisionを前段のfantasyへ結び付けます。この計算は[KGの上流実装][kg]と[multi-stepの上流実装][tree]が担当します。
robotorchanの[optimizer_compatibility.py][compat]はこの構造を持つacquisitionと候補optimizerの対応を判定し、
fantasy計算自体は実装しません。

[kg]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/knowledge_gradient.py
[tree]: https://github.com/pytorch/botorch/blob/v0.18.1/botorch/acquisition/multi_step_lookahead.py
[compat]: ../../../src/robotorchan/acquisition/optimizer_compatibility.py

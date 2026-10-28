# Lookahead 獲得関数

BoTorch 標準 API を直接使います。モデルが BoTorch の契約を満たすため、これらの獲得関数を wrapper 化・再公開しません。

## Knowledge Gradient

1 段先の情報価値を最適化するには `qKnowledgeGradient` を使います。

```python
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

acqf = qKnowledgeGradient(model=model, num_fantasies=64)
```


獲得関数へ直接渡す Tensor には、実際の q 候補と KG が要求する fantasy 点を含めます。通常は、この拡張 Tensor を手で作らず BoTorch 専用の KG 最適化機能を使ってください。

`optimize_acqf` は qKG を one-shot と認識し、拡張 q-batch 全体の初期値を生成します。robotorchan はこの初期化を置き換えません。明示的な `batch_initial_conditions` も `acqf.get_augmented_q_batch_size(q)` 行が必要です。最終結果には `extract_candidates` 後の要求候補だけが含まれます。

## Multi-step lookahead

非近視眼的な意思決定木には `qMultiStepLookahead` を使います。

```python
import torch
from botorch.acquisition.multi_step_lookahead import qMultiStepLookahead
from botorch.sampling.normal import SobolQMCNormalSampler

samplers = [
    SobolQMCNormalSampler(sample_shape=torch.Size([16])),
    SobolQMCNormalSampler(sample_shape=torch.Size([16])),
]
acqf = qMultiStepLookahead(
    model=model,
    batch_sizes=[1, 1],
    samplers=samplers,
)
```


`batch_sizes` は将来の段階、`samplers` は fantasy の分岐を定めます。horizon と fantasy 数で拡張 batch が急増するため、追加の意思決定価値が計算費用を正当化する場合に使います。

qKG と異なり、専用初期化は自動選択されません。通常の初期化器には公開 `q` が渡る一方、獲得関数は拡張木全体を評価します。`ic_generator` に `gen_augmented_one_shot_initial_conditions` を指定するか、`acqf.get_augmented_q_batch_size(q)` 行の初期値を明示してください。この helper は BoTorch `gen_batch_initial_conditions` を拡張 batch に適用し、別のサンプリング規則は導入しません。

同じ helper は、標準 box・線形制約の初期化を使い、拡張 batch が必要で専用初期化器を持たない custom `OneShotAcquisitionFunction` にも使えます。qKG のような専用経路では標準の初期化器を維持してください。

## 適用範囲

独自の lookahead 抽象化は追加せず、BoTorch 経路との接続を検証します。多目的・多忠実度の lookahead はそれぞれの専用経路を参照してください。[Lookahead 理論](../theory/acquisition/06_lookahead.md)に手法間の関係を示します。

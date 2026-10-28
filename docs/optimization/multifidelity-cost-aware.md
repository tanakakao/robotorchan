# 多忠実度・コスト考慮の獲得関数

BoTorch 標準の仕組みを使い、alias は追加しません。

## Multi-fidelity Knowledge Gradient

連続 fidelity には `qMultiFidelityKnowledgeGradient`（MF-KG）を使います。情報価値を、目標 fidelity への射影および任意のコスト効用と組み合わせます。

```python
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.models.cost import AffineFidelityCostModel

cost_model = AffineFidelityCostModel(
    fidelity_weights={fidelity_dim: 1.0},
    fixed_cost=0.1,
)
cost_utility = InverseCostWeightedUtility(cost_model=cost_model)

acqf = qMultiFidelityKnowledgeGradient(
    model=model,
    num_fantasies=64,
    current_value=current_value,
    cost_aware_utility=cost_utility,
    project=project_to_target_fidelity,
)
```


射影は最終意思決定の価値を測る目標 fidelity へ候補を写す、問題定義の一部です。汎用 factory の内部へ隠しません。

## コストモデル

fidelity のアフィン関数として費用が既知なら `AffineFidelityCostModel` を使います。学習する場合は正の費用 surrogate と `InverseCostWeightedUtility` を組み合わせます。逆重み付けに使う cost objective は厳密に正である必要があり、正のスケールの予測か明示的な objective・transform を用意してください。

## 離散 fidelity

許容 fidelity が有限集合なら、獲得関数は標準のまま、BoTorch の mixed・discrete 最適化などで候補を制限します。カテゴリ fidelity を誤って無制約の連続変数へ緩和しないでください。

## 実行検証の範囲

- `SingleTaskMultiFidelityGP` と `MixedSingleTaskMultiFidelityGP`。
- qMFKG の目標 fidelity 射影と、`current_value` 用の目標 fidelity 上の incumbent 最適化。
- `AffineFidelityCostModel` と `InverseCostWeightedUtility`。
- 学習済み robotorchan cost surrogate による逆コスト重み。
- 連続 qMFKG の評価と mixed qMFKG の候補生成。

実験候補の fidelity 自体は最適化変数です。最終価値を測る射影によって、評価候補を暗黙に目標 fidelity へ固定しません。

MF-KG は one-shot です。BoTorch が要求候補と fantasy 行を含む拡張 batch を最適化し、実候補を抽出します。自動初期化は上流へ委譲します。明示的な `batch_initial_conditions` の q 次元は、公開 q ではなく `acqf.get_augmented_q_batch_size(q)` に合わせてください。

## Multi-fidelity MES

MF-MES は現在の実行検証範囲外です。candidate set、fidelity、コスト合成、optimizer の別途検証が必要です。MF-KG で検証済みの非近視眼的・コスト考慮経路とは区別し、将来の拡張として扱います。

## 責務の境界

モデル選択と fidelity 次元の宣言は surrogate 側の責務です。多目的・多忠実度 lookahead と MF-MES を実行検証済み経路として案内しません。[多忠実度の理論](../theory/acquisition/11_multifidelity_cost_aware.md)を参照してください。

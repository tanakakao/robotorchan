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


## 学習型コストモデルと予算管理

コストが未知なら、既存のrobotorchan `SingleTaskGP`をコストの
surrogateとして学習し、BoTorchの`InverseCostWeightedUtility`に渡します。
負の予測コストによる不正な重み付けを避けるため、実測コストが厳密に正なら
`log(cost)`を学習し、`GenericMCObjective`で`exp`に戻す方法が使えます。

`use_mean=True`とこの変換の組み合わせは
`exp(E[log(cost)])`であり、`E[cost]`そのものではありません。
モデルの不確実性まで考慮する場合は、cost objectiveとMC評価を
目的に合わせて設計してください。コストがゼロになり得る場合は、
log変換前の扱いを別途定義する必要があります。

[実行可能な学習型コスト例](../../examples/learned_multifidelity_cost.py)では、
cost GPの学習、正値化したコスト効用、Phase 2の`CostBudget`による
予測費用の予約と実測費用の精算を接続します。
この例はcost utilityの動作を示すものであり、
MF-KGの全closed-loop最適化はPhase 4の対象です。


## MF-KGの逐次Closed-loop（Phase 4）

[実行例](../../examples/mfkg_cost_budget_loop.py)は、
BoTorch標準のMF-KGと`CostBudget`を接続し、
各反復でモデルの再学習、target fidelityでの`current_value`計算、
評価候補の最適化、費用予約、評価、実費精算を行います。

- 評価候補のfidelityは`[0.5, 1.0]`で最適化し、
  `project_to_target`は最終価値の評価にだけ用います。
- 評価費用は既知の`AffineFidelityCostModel`を使用します。
  Phase 3の学習型cost surrogateへの差し替えは可能ですが、
  本例ではobjective側の逐次更新を明確にするため固定しています。
- 初期観測費用は予算計算の対象外です。必要なら
  `CostBudget.record`で初期費用を計上してください。
- 予測費用が残予算を超えた場合は停止します。
  別の安い候補の探索や候補集合の予算制約付き最適化は行いません。
- 評価例外では予約を取り消します。評価済みだが実費が不明な場合は
  利用側で実験記録と整合させてください。
- この例は同期・逐次の`q=1`です。非同期の同時予約や
  複数候補のjoint batch運用は対象外です。


## MF-MES実装可否の整理（Phase 5）

MF-MES（Multi-Fidelity Max-value Entropy Search）は、最終的なtarget
fidelityでの最適値について、候補`(x, s)`を評価することで得られる
情報量をコストと合わせて評価する手法です。通常のMESの
`qMaxValueEntropy`を多忠実度モデルに渡しただけでは、
target fidelityの最適値に関する情報量という要件を満たすとは限りません。

### 既存実装と不足点

| 項目 | 現状 | 判断 |
| --- | --- | --- |
| 通常のMES / GIBBON | BoTorchの`qMaxValueEntropy` / `qLowerBoundMaxValueEntropy`を統合テスト済み | 再利用 |
| Multi-Fidelity GP | robotorchanのモデルとBoTorchのMF-KG連携を検証済み | 再利用 |
| Cost model / budget | 既知・学習型costと`CostBudget`の接続例あり | 再利用 |
| MF-MES acquisition | robotorchan内に実行検証済みの専用経路なし | 未対応 |
| MF-MES候補生成・閉ループ | 専用の候補生成テストなし | 未対応 |

### 実装判断の順序

1. 使用するBoTorchバージョンに、target-fidelity optimumの
   サンプリングとfidelity間の条件付き情報量を扱う公開MF-MES APIが
   存在するか確認する。**本Phaseでは上流APIの存在を確定していない。**
2. 公開APIが利用できるなら、robotorchan独自クラスを追加せず、
   fidelity projection、target fidelity上のcandidate set、
   cost utility、optimizerを組み合わせて小さな統合テストを行う。
3. 公開APIがない場合、通常のMESにcost重みを付けただけの実装を
   MF-MESと呼ばない。必要な情報量近似、fidelity間相関の条件付け、
   計算量を明示した別途のアルゴリズム設計が必要となる。
4. 追加アルゴリズムの必要性が確認されるまでは、
   実行検証済みのcost-aware MF-KGを推奨経路として維持する。

### 最小限の検証条件

- 評価fidelityとtarget fidelityが異なるケースを含める。
- candidate setのfidelityと最適値の定義を混同しない。
- コストが正で有限であること、acquisition値が有限であることを確認する。
- `q=1`の候補提案までを先に検証し、batch・mixed・非同期は
  初回実装の必須条件にしない。
- MF-KGとの性能比較は、実行可能なMF-MES経路が確立してから行う。

**Phase 5結論：現時点でrobotorchanのMF-MES対応完了とは判定しない。**
公開BoTorch APIの確認と最小の統合実験が次の実装着手条件です。
MF-KGのPhase 4までの成果とは独立した拡張候補として扱います。

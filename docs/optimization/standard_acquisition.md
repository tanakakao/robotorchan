# 標準の単一目的獲得関数

BoTorch 標準 class を直接使い、重複実装や再公開を行いません。

## 推奨する既定の選択

改善量に基づく BO では、数値的に安定した log-space 版を推奨します。

- `LogExpectedImprovement`：q=1 の解析的期待改善。
- `qLogExpectedImprovement`：Monte Carlo の batch 期待改善。
- `qLogNoisyExpectedImprovement`：ノイズのある観測。

従来の `qExpectedImprovement` には既知の数値問題があり、BoTorch も log 版を推奨します。noisy 改善量でも同じ方針です。再現実験で必要なら従来版を BoTorch から使えますが、既定の推奨ではありません。

その他の標準 API も直接使います。

- `LogProbabilityOfImprovement`：改善確率。
- `UpperConfidenceBound`・`qUpperConfidenceBound`：信頼境界による探索。
- `qSimpleRegret`：posterior utility・simple regret 型の選択。制約なし単一目的の推奨と q-batch 最適化に対応する capability を登録しています。

## 使用例

```python
from botorch.acquisition.logei import qLogExpectedImprovement

acqf = qLogExpectedImprovement(model=model, best_f=train_Y.max())
```


返り値は BoTorch 獲得関数であり、`botorch.optim.optimize_acqf` へ直接渡せます。

## モデルとの互換性

互換性は wrapper の有無ではなくモデル・posterior の契約で決まります。標準的な GP モデルは BoTorch 互換の posterior を公開し、`SingleTaskGP` と解析的・MC 獲得関数の統合テストがあります。

non-GP の経験 ensemble は異なります。解析的 Gaussian 獲得関数はその経験 posterior に適用できないため、獲得関数アーキテクチャにある検証 helper を使ってください。

多目的、多忠実度、情報理論、lookahead、Active Learning、level-set は専用ガイドに分けています。[選択の理論](../theory/acquisition/12_selection_guide.md)も参照してください。

## Causal BO Phase 4: intervention-set selection

For known DAGs, `propose_causal_intervention` compares **fitted**
`InterventionGPSurrogate` instances from distinct, actually performed
hard-intervention sets. Each set is optimized in its own continuous bounds,
using BoTorch-native analytic `LogExpectedImprovement` and robotorchan
`optimize_acqf(q=1)`. No causal graph discovery is performed.

All sets use the same incumbent `best_f`, pooled from their observed
interventional outcomes. For a fixed positive set cost `c(S)`, the ranking is

```text
log_utility(S) = log(EI_S(x_S*)) - log(c(S))
```

Since the cost is constant within a set, the same `x_S*` maximizes LogEI
and cost-adjusted LogEI. **Do not divide LogEI by cost**: that is not
equivalent to `log(EI / cost)`. Missing costs default to 1.

```python
from robotorchan.causal import propose_causal_intervention

proposal = propose_causal_intervention(
    [surrogate_x, surrogate_z],
    {"X": (0.0, 1.0), "Z": (0.0, 1.0)},
    intervention_costs={frozenset({"X"}): 2.0},
    optimizer="botorch",
)
print(proposal.intervention.values, proposal.log_utility)
```

Both surrogates must have been fitted using `fit()`, refer to the **same**
graph instance and outcome, and represent distinct intervention sets.
Only intervened training outcomes contribute to the common incumbent;
observation-only records are never interpreted as `do` data. Minimization
is supported with `direction="minimize"`. The values in the returned
`HardIntervention` use the corresponding surrogate's canonical DAG order.

Scope: continuous inputs, one intervention at a time (`q=1`), a single
outcome, fixed per-set positive costs, and bounds on candidate inputs.
Candidate-dependent cost models, cross-set covariance, constrained/mixed
intervention variables, batch `q>1` and a full ask/tell loop remain future
work. This is an empirical intervention-policy selector rather than a
general observational causal-effect estimator.

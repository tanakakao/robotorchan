# Input perturbation contract

この文書は nominal candidate に decision-time perturbation を与え、scenario risk を
集約する workflow の開発契約を定義します。uncertain-input surrogate modelling や
observation-noise robustness とは別の capability として扱います。

## Semantic pipeline

標準 workflow は次です。

```text
nominal raw X
 -> perturb allowed raw dimensions
 -> model-owned transform / reduction
 -> posterior
 -> posterior samples
 -> scenario-aware risk objective
 -> MC acquisition
 -> candidate optimization
```

reduced / learned representation model でも、物理的 perturbation は原則として raw design
space に適用し、その後 model-owned transform を通します。latent-space offset を
物理的不確かさとして暗黙に扱いません。

## Protected dimensions

perturbation の対象は design / environmental continuous coordinates です。次の structural
coordinates は default で保護します。

- categorical
- task
- fidelity
- context
- hierarchy
- その他 discrete structural coordinates

Mixed + MultiTask / MultiFidelity では protected dimensions の union を守ります。
MultiTask の task feature、MultiFidelity の fidelity feature を perturb してはいけません。

## Certification rule

constructor signature、inheritance、posterior capability metadata だけでは input
perturbation 対応を claim しません。certification は runtime evidence に基づきます。

代表的な certification path は以下を実行します。

1. raw candidate への perturbation
2. protected-coordinate invariance
3. posterior evaluation / sampling
4. scenario-risk aggregation
5. compatible MC acquisition
6. candidate optimization

structured posterior、variational / fully Bayesian、ModelList、high-dimensional encoder などは
各 family 固有の shape / sampler semantics を追加で検証します。

## Capability boundaries

- exact GP: standard certification candidate
- Mixed: continuous dimensions のみ perturb
- MultiTask: task feature を保護
- Kronecker: output/task structure を保持
- MultiFidelity: fidelity coordinates を保護
- PCA / PLS / random projection: raw design space で perturb 後に reduction
- AE / VAE / joint encoder: raw-space perturbation と model-owned transform を検証
- robust observation models: observation robustness と perturbation risk を分離
- contextual / hierarchical: structural coordinates を保護
- Pairwise: preference likelihood 用の別設計が必要
- deterministic tree / boosting: scenario-evaluation adapter が必要
- NGBoost: distributional scenario aggregation と BoTorch InputPerturbation contract を
  同一視しない
- uncertain-input models: separate mechanism。個別に E2E certification が必要

## Risk semantics

risk aggregator が存在することだけでは model compatibility を意味しません。
expectation、mean-variance、worst-case、VaR、CVaR、signal-to-noise 等は、
posterior/scenario tensor の軸を正しく解釈する runtime path と組み合わせて検証します。

## Maintenance rule

model family を certified と記録する場合、その claim を支える executable test を
同時に保持します。新しい structural dimension や transform を追加した場合は、
protected-coordinate contract と perturbation ordering を再監査します。

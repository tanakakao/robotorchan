# Phase 3: Intervention GP Surrogate

Phase 3 uses the existing `robotorchan.models.standard.single_task.SingleTaskGP`.
It does **not** infer a causal graph, identify causal effects from observational
data, fit GP structural equations, or implement CBO candidate selection.

## Meaning of the GP

For a fixed intervention set `S`, the training pairs are

```text
X = values explicitly assigned by do(X_S = x_S)
Y = outcome measured after that intervention
```

The GP approximates the response under **performed hard interventions** on
exactly `S`. Observational-only records, and records intervening on different
sets, are deliberately excluded. Observational conditionals `E[Y | X=x]`
must not be mistaken for interventional expectations `E[Y | do(X=x)]`.
Interpretation of predicted responses assumes the trials follow a comparable
intervention regime, including their background conditions.

## API example

```python
import torch
from robotorchan.causal import (
    CausalGraph, CausalObservation, HardIntervention, InterventionGPSurrogate,
)

graph = CausalGraph({"X": (), "Y": ("X",)})
records = [
    CausalObservation(
        {"X": x, "Y": x * x},
        HardIntervention({"X": x}),
    )
    for x in (0.0, 0.3, 0.6, 1.0)
]
surrogate = InterventionGPSurrogate(graph, "Y", ("X",), records)
surrogate.fit()
posterior = surrogate.posterior(torch.tensor([[0.4]], dtype=torch.double))
print(posterior.mean, posterior.variance)
```

`surrogate.model` is the native robotorchan `SingleTaskGP`, which inherits
BoTorch's posterior contract and can be passed directly to suitable BoTorch
acquisition functions in later phases.

Use **one surrogate per intervention set**. Input columns are ordered according
to `graph.topological_order`, filtered to the intervention nodes, regardless
of the order of `intervention_nodes` passed by the caller.

At least two matching intervention records are required. The GP uses the
existing inferred-noise `SingleTaskGP` defaults. No cross-intervention-set
sharing, observational-data adjustment, selection of intervention sets,
intervention cost weighting, or full CBO optimizer is implemented in this phase.

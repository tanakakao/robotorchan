"""Runtime inputs retained when usage examples were removed from Guide."""

import torch

from robotorchan.optim.trust_region.turbo import TuRBOState, update_turbo_state

state = TuRBOState(
    dim=2,
    batch_size=1,
    length=0.4,
    success_tolerance=2,
    failure_tolerance=2,
)
state = update_turbo_state(state, torch.tensor([0.5], dtype=torch.double))
assert state.length == 0.4
assert state.success_counter == 1
state = update_turbo_state(state, torch.tensor([0.7], dtype=torch.double))
assert state.length == 0.8
assert state.success_counter == 0
state = update_turbo_state(state, torch.tensor([0.6], dtype=torch.double))
assert state.length == 0.8
assert state.failure_counter == 1
state = update_turbo_state(state, torch.tensor([0.6], dtype=torch.double))
assert state.length == 0.4
assert state.failure_counter == 0
assert state.best_value == 0.7
assert state.observed_best_value == 0.7

from robotorchan.optim.capabilities import (
    BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
    BOTORCH_OPTIMIZER_CAPABILITIES,
    OptimizerCapabilities,
)


def test_optimizer_capabilities_are_immutable() -> None:
    capabilities = OptimizerCapabilities()

    try:
        capabilities.continuous = False
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("OptimizerCapabilities must be immutable.")


def test_botorch_capabilities_match_native_optimizer_contract() -> None:
    capabilities = BOTORCH_OPTIMIZER_CAPABILITIES

    assert capabilities.continuous
    assert capabilities.requires_grad
    assert capabilities.linear_inequality_constraints
    assert capabilities.linear_equality_constraints
    assert capabilities.nonlinear_inequality_constraints
    assert capabilities.interpoint_nonlinear_constraints
    assert capabilities.q_batch
    assert capabilities.sequential
    assert capabilities.fixed_features
    assert not capabilities.mixed


def test_botorch_mixed_capabilities_are_explicit() -> None:
    capabilities = BOTORCH_MIXED_OPTIMIZER_CAPABILITIES

    assert capabilities.continuous
    assert capabilities.integer
    assert capabilities.categorical
    assert capabilities.mixed
    assert capabilities.q_batch
    assert capabilities.fixed_features
    assert capabilities.nonlinear_inequality_constraints
    assert not capabilities.interpoint_nonlinear_constraints

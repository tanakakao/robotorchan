from robotorchan.optim.capabilities import (
    BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
    BOTORCH_OPTIMIZER_CAPABILITIES,
    DIFFERENTIAL_EVOLUTION_OPTIMIZER_CAPABILITIES,
    PSO_OPTIMIZER_CAPABILITIES,
    ConstraintHandling,
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


def test_constraint_handling_distinguishes_native_and_feasibility_first() -> None:
    botorch = BOTORCH_OPTIMIZER_CAPABILITIES.constraint_handling
    de = DIFFERENTIAL_EVOLUTION_OPTIMIZER_CAPABILITIES.constraint_handling

    assert botorch.linear_inequality is ConstraintHandling.NATIVE
    assert botorch.linear_equality is ConstraintHandling.NATIVE
    assert botorch.nonlinear_inequality is ConstraintHandling.NATIVE
    assert botorch.interpoint_nonlinear is ConstraintHandling.NATIVE
    assert de.linear_inequality is ConstraintHandling.FEASIBILITY_FIRST
    assert de.linear_equality is ConstraintHandling.FEASIBILITY_FIRST
    assert de.nonlinear_inequality is ConstraintHandling.FEASIBILITY_FIRST
    assert de.interpoint_nonlinear is ConstraintHandling.FEASIBILITY_FIRST


def test_unsupported_constraint_handling_is_explicit() -> None:
    capabilities = OptimizerCapabilities()

    assert capabilities.constraint_handling.linear_inequality is ConstraintHandling.UNSUPPORTED
    assert capabilities.constraint_handling.linear_equality is ConstraintHandling.UNSUPPORTED
    assert capabilities.constraint_handling.nonlinear_inequality is ConstraintHandling.UNSUPPORTED
    assert capabilities.constraint_handling.interpoint_nonlinear is ConstraintHandling.UNSUPPORTED



def test_pso_capabilities_do_not_overstate_categorical_mixed_support() -> None:
    capabilities = PSO_OPTIMIZER_CAPABILITIES

    assert capabilities.continuous
    assert capabilities.integer
    assert not capabilities.categorical
    assert not capabilities.mixed

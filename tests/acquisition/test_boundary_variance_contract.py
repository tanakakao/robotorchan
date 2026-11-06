import inspect

from robotorchan.acquisition import qBoundaryVariance


def test_boundary_variance_does_not_advertise_unused_beta() -> None:
    signature = inspect.signature(qBoundaryVariance.__init__)

    assert "target" in signature.parameters
    assert "output_index" in signature.parameters
    assert "beta" not in signature.parameters

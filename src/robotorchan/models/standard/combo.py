"""Graph-diffusion Gaussian process surrogate on the binary hypercube.

The kernel is a tensor product of normalized heat kernels on two-node graphs.
This is a binary-domain COMBO-style surrogate, not a general categorical
graph product or a full implementation of the COMBO inference algorithm.
"""

from __future__ import annotations

import torch
from gpytorch.kernels import Kernel
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import softplus

from robotorchan.models.standard.single_task import SingleTaskGP


class BinaryGraphDiffusionKernel(Kernel):
    """Learnable product-graph heat kernel for binary-valued features.

    On each two-node graph, the normalized heat kernel has diagonal one
    and off-diagonal tanh(t), where t > 0 is a learned diffusion time.
    The Cartesian-product graph kernel is the product over dimensions.
    """

    def __init__(self, input_dim: int) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        super().__init__()
        self.input_dim = input_dim
        self.raw_diffusion = Parameter(torch.zeros(input_dim))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if x1.shape[-1] != self.input_dim or x2.shape[-1] != self.input_dim:
            raise ValueError("Kernel inputs must match input_dim.")
        if diag:
            if x1.shape[-2] != x2.shape[-2]:
                raise ValueError("Diagonal evaluation requires equal point counts.")
            mismatches = (x1 != x2).to(dtype=x1.dtype)
        else:
            mismatches = (x1.unsqueeze(-2) != x2.unsqueeze(-3)).to(dtype=x1.dtype)
        diffusion = softplus(self.raw_diffusion) + 1e-6
        log_correlation = torch.log(torch.tanh(diffusion))
        return torch.exp((mismatches * log_correlation).sum(dim=-1))


class COMBOSurrogate(SingleTaskGP):
    """BoTorch-compatible GP with a binary graph-diffusion kernel.

    Supports standard SingleTaskGP fit, posterior, and acquisition APIs.
    Phase 7 is restricted to binary inputs and single-output regression.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
    ) -> None:
        if train_X.ndim != 2 or train_X.shape[-1] < 1:
            raise ValueError("train_X must have shape [n, d] with d >= 1.")
        if train_Y.shape != (train_X.shape[0], 1):
            raise ValueError("train_Y must have shape [n, 1].")
        if not train_X.is_floating_point():
            raise ValueError("train_X must be floating point.")
        if not bool(((train_X == 0) | (train_X == 1)).all()):
            raise ValueError("COMBO currently requires binary (0/1) inputs.")
        kernel = BinaryGraphDiffusionKernel(train_X.shape[-1]).to(
            dtype=train_X.dtype, device=train_X.device
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=kernel,
        )

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ):
        if X.ndim < 2 or X.shape[-1] != self.train_inputs[0].shape[-1]:
            raise ValueError("X must have shape [..., q, d].")
        if not bool(((X == 0) | (X == 1)).all()):
            raise ValueError("COMBO currently requires binary (0/1) query inputs.")
        return super().posterior(
            X,
            output_indices=output_indices,
            observation_noise=observation_noise,
            posterior_transform=posterior_transform,
        )

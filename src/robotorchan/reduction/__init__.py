"""Dimensionality-reduction utilities for robotorchan models."""

from robotorchan.reduction.base import InputReducer, OutputReducer, ReducerNotFittedError, TensorReducer
from robotorchan.reduction.input import PCAInputReducer, PLSInputReducer, RandomProjectionInputReducer
from robotorchan.reduction.neural import (
    AutoEncoderInputReducer,
    SupervisedAutoEncoderInputReducer,
    SupervisedVAEInputReducer,
    VAEInputReducer,
)
from robotorchan.reduction.output import LinearOutputPosterior, OutputPCAReducer, OutputPLSReducer

__all__ = [
    "PCAInputReducer",
    "PLSInputReducer",
    "VAEInputReducer",
    "AutoEncoderInputReducer",
    "InputReducer",
    "LinearOutputPosterior",
    "OutputPCAReducer",
    "OutputPLSReducer",
    "OutputReducer",
    "RandomProjectionInputReducer",
    "ReducerNotFittedError",
    "SupervisedAutoEncoderInputReducer",
    "SupervisedVAEInputReducer",
    "TensorReducer",
]

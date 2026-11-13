"""Standard binary classification surrogate models."""

from robotorchan.models.classification.binary.standard.mixed_spectral_mixture import (
    MixedSpectralMixtureBinaryGPClassifier,
)
from robotorchan.models.classification.binary.standard.multi_fidelity import (
    MixedMultiFidelityBinaryGPClassifier,
    MultiFidelityBinaryGPClassifier,
)
from robotorchan.models.classification.binary.standard.multitask import (
    KroneckerMultiTaskBinaryGPClassifier,
    MultiTaskBinaryGPClassifier,
)
from robotorchan.models.classification.binary.standard.single_task import (
    BinarySingleTaskGPClassifier,
    MixedBinarySingleTaskGPClassifier,
)
from robotorchan.models.classification.binary.standard.spectral_mixture import (
    SpectralMixtureBinaryGPClassifier,
)

__all__ = [
    "BinarySingleTaskGPClassifier",
    "KroneckerMultiTaskBinaryGPClassifier",
    "MixedBinarySingleTaskGPClassifier",
    "MixedMultiFidelityBinaryGPClassifier",
    "MixedSpectralMixtureBinaryGPClassifier",
    "MultiFidelityBinaryGPClassifier",
    "MultiTaskBinaryGPClassifier",
    "SpectralMixtureBinaryGPClassifier",
]

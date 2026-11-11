"""Candidate-domain contracts shared by acquisition optimizers."""

from robotorchan.optim.domains.combinatorial import hamming_distance, hamming_neighbors
from robotorchan.optim.domains.variable_space import MixedVariableSpace

__all__ = ["MixedVariableSpace", "hamming_distance", "hamming_neighbors"]

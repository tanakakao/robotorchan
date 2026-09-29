"""Acquisition-function optimization and search-strategy interfaces."""

from robotorchan.optim.base import SearchResult, SearchStrategy
from robotorchan.optim.benchmark import BenchmarkResult, benchmark_optimizer, benchmark_optimizers
from robotorchan.optim.capabilities import (
    BOTORCH_MIXED_OPTIMIZER_CAPABILITIES,
    BOTORCH_OPTIMIZER_CAPABILITIES,
    CMAES_OPTIMIZER_CAPABILITIES,
    DIFFERENTIAL_EVOLUTION_OPTIMIZER_CAPABILITIES,
    GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
    HYBRID_OPTIMIZER_CAPABILITIES,
    MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES,
    NSGA2_OPTIMIZER_CAPABILITIES,
    OPTIMIZER_CAPABILITIES,
    PSO_OPTIMIZER_CAPABILITIES,
    SAMPLING_OPTIMIZER_CAPABILITIES,
    TORCH_OPTIMIZER_CAPABILITIES,
    ConstraintHandling,
    ConstraintHandlingCapabilities,
    OptimizerCapabilities,
    get_optimizer_capabilities,
)
from robotorchan.optim.constraints import (
    CandidateConstraints,
    LinearConstraint,
    NonlinearConstraint,
    NonlinearConstraintCallable,
)
from robotorchan.optim.cross_cutting import apply_fixed_features, optimize_acqf_sequential
from robotorchan.optim.dispatch import OptimizerName, optimize_acqf
from robotorchan.optim.embedding import (
    ALEBOStrategy,
    BAxUSState,
    BAxUSStrategy,
    BAxUSThompsonSamplingStrategy,
    HeSBOStrategy,
    REMBOStrategy,
    update_baxus_state,
)
from robotorchan.optim.initializers import gen_augmented_one_shot_initial_conditions
from robotorchan.optim.latent import (
    LatentReconstruction,
    LatentSpaceStrategy,
    PCAReconstruction,
    RandomProjectionReconstruction,
)
from robotorchan.optim.mixed import MixedSpaceStrategy
from robotorchan.optim.mixed_one_shot import optimize_mixed_one_shot_acqf
from robotorchan.optim.original import OriginalSpaceStrategy
from robotorchan.optim.random import RandomSearchStrategy
from robotorchan.optim.sobol import SobolSearchStrategy
from robotorchan.optim.tree import TreeEnsembleSearchStrategy
from robotorchan.optim.trust_region import TuRBOState, TuRBOStrategy, update_turbo_state
from robotorchan.optim.variable_space import MixedVariableSpace

__all__ = [
    "BOTORCH_MIXED_OPTIMIZER_CAPABILITIES",
    "BOTORCH_OPTIMIZER_CAPABILITIES",
    "CMAES_OPTIMIZER_CAPABILITIES",
    "DIFFERENTIAL_EVOLUTION_OPTIMIZER_CAPABILITIES",
    "GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES",
    "HYBRID_OPTIMIZER_CAPABILITIES",
    "MIXED_GENETIC_ALGORITHM_OPTIMIZER_CAPABILITIES",
    "NSGA2_OPTIMIZER_CAPABILITIES",
    "OPTIMIZER_CAPABILITIES",
    "PSO_OPTIMIZER_CAPABILITIES",
    "SAMPLING_OPTIMIZER_CAPABILITIES",
    "TORCH_OPTIMIZER_CAPABILITIES",
    "ALEBOStrategy",
    "BAxUSState",
    "BAxUSStrategy",
    "BAxUSThompsonSamplingStrategy",
    "BenchmarkResult",
    "CandidateConstraints",
    "ConstraintHandling",
    "ConstraintHandlingCapabilities",
    "HeSBOStrategy",
    "LatentReconstruction",
    "LatentSpaceStrategy",
    "LinearConstraint",
    "MixedSpaceStrategy",
    "MixedVariableSpace",
    "NonlinearConstraint",
    "NonlinearConstraintCallable",
    "OptimizerCapabilities",
    "OptimizerName",
    "OriginalSpaceStrategy",
    "PCAReconstruction",
    "REMBOStrategy",
    "RandomProjectionReconstruction",
    "RandomSearchStrategy",
    "SearchResult",
    "SearchStrategy",
    "SobolSearchStrategy",
    "TreeEnsembleSearchStrategy",
    "TuRBOState",
    "TuRBOStrategy",
    "apply_fixed_features",
    "benchmark_optimizer",
    "benchmark_optimizers",
    "gen_augmented_one_shot_initial_conditions",
    "get_optimizer_capabilities",
    "optimize_acqf",
    "optimize_acqf_sequential",
    "optimize_mixed_one_shot_acqf",
    "update_baxus_state",
    "update_turbo_state",
]

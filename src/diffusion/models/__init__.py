"""
Public model API for diffusion score models.

Canonical active workflow:
- Register models/wrappers in diffusion.models.registry
- Instantiate models using diffusion.models.create_model

"""

# Initialize the canonical model registry on import.
from . import registry as _registry  # noqa: F401

from .base import (
    ScoreModel,
    WrappedScoreModel,
    wrap_existing_model,
    ChannelSqueezeWrapper,
    validate_score_output,
)
from .mlp import (
    MLPScoreModel,
    ResidualMLPScoreModel,
    TimeConditionedMLPScoreModel,
    LayerNormMLPScoreModel,
)
from .matrix import TraceMLPScoreModel
from .temporal import UNet1DTimeScoreModel, TemporalUnetScoreModel
from .wrappers import (
    DiffusionScalingWrapper,
    ResidualWrapper,
    LogTimeWrapper,
    VParamWrapper,
    SequentialWrapper,
)
from .model_factory import create_model

__all__ = [
    "ScoreModel",
    "WrappedScoreModel",
    "wrap_existing_model",
    "ChannelSqueezeWrapper",
    "validate_score_output",
    "MLPScoreModel",
    "ResidualMLPScoreModel",
    "TimeConditionedMLPScoreModel",
    "LayerNormMLPScoreModel",
    "TraceMLPScoreModel",
    "UNet1DTimeScoreModel",
    "TemporalUnetScoreModel",
    "DiffusionScalingWrapper",
    "ResidualWrapper",
    "LogTimeWrapper",
    "VParamWrapper",
    "SequentialWrapper",
    "create_model",
]

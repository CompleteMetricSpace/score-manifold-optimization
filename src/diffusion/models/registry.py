"""
Central registry for all model architectures and wrapper configurations.

Canonical model extension path for active code:
- Add or update model classes under diffusion.models.*
- Register them here using register_model/register_wrapper_config
- Instantiate through diffusion.models.model_factory.create_model
"""

from .model_factory import register_model, register_wrapper_config

# ============================================================================
# BASE MODEL REGISTRATION
# ============================================================================

from .mlp import MLPScoreModel, ResidualMLPScoreModel, LayerNormMLPScoreModel
from .matrix import TraceMLPScoreModel
from .temporal import UNet1DTimeScoreModel, TemporalUnetScoreModel

register_model("MLPScoreModel", MLPScoreModel)
register_model("ResidualMLPScoreModel", ResidualMLPScoreModel)
register_model("LayerNormMLPScoreModel", LayerNormMLPScoreModel)
register_model("TraceMLPScoreModel", TraceMLPScoreModel)
register_model("UNet1DTimeScoreModel", UNet1DTimeScoreModel)
register_model("TemporalUnetScoreModel", TemporalUnetScoreModel)

# ============================================================================
# WRAPPED MODEL CONFIGURATIONS
# ============================================================================

# ------------------------------------------------------------------
# MLP Family (VectorSpace)
# ------------------------------------------------------------------

register_wrapper_config(
    name="mlp",
    base_model="MLPScoreModel",
    wrappers=[],
    base_kwargs={}
)

register_wrapper_config(
    name="mlp_diff_scale",
    base_model="MLPScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-6})]
)

register_wrapper_config(
    name="mlp_diff_scale_sq",
    base_model="MLPScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "var", "clip_eps": 1e-6})]
)

register_wrapper_config(
    name="mlp_diff_scale_res",
    base_model="MLPScoreModel",
    wrappers=[
        ("residual", {"positive": True}),
        ("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-6}),
    ]
)


register_wrapper_config(
    name="mlp_vparam",
    base_model="MLPScoreModel",
    wrappers=[
        ("vparam", {"clip_eps": 1e-5})
    ]
)

# TraceMLPTimeModule Family (MatrixSpace)
# ------------------------------------------------------------------

register_wrapper_config(
    name="TraceMLPTimeModule",
    base_model="TraceMLPScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "var", "clip_eps": 1e-6})]
)

register_wrapper_config(
    name="TraceMLPTimeModuleV",
    base_model="TraceMLPScoreModel",
    wrappers=[("vparam", {"clip_eps": 1e-6})]
)

register_wrapper_config(
    name="TraceMLPTimeModule2",
    base_model="TraceMLPScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-6})]
)

register_wrapper_config(
    name="TraceMLPTimeModule3",
    base_model="TraceMLPScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-6})]
)

register_wrapper_config(
    name="TraceMLPTimeModuleResNet",
    base_model="TraceMLPScoreModel",
    wrappers=[
        ("diffusion_scaling", {"mode": "var", "clip_eps": 1e-4}),
        ("residual", {"positive": True}),
    ]
)

register_wrapper_config(
    name="TraceMLPTimeModuleResNet2",
    base_model="TraceMLPScoreModel",
    wrappers=[
        ("residual", {"positive": True}),
        ("diffusion_scaling", {"mode": "var", "clip_eps": 1e-8}),
    ]
)

register_wrapper_config(
    name="TraceMLPTimeModuleResNet3",
    base_model="TraceMLPScoreModel",
    wrappers=[
        ("residual", {"positive": True}),
        ("diffusion_scaling", {"mode": "var", "clip_eps": 1e-4}),
    ]
)

register_wrapper_config(
    name="TraceMLPTimeModuleResNet4",
    base_model="TraceMLPScoreModel",
    wrappers=[
        ("log_time", {}),
        ("residual", {"positive": True}),
        ("diffusion_scaling", {"mode": "var", "clip_eps": 1e-4}),
    ]
)

register_wrapper_config(
    name="TraceMLPTimeModuleTimeEmbedding",
    base_model="TraceMLPScoreModel",
    wrappers=[
        ("residual", {"positive": True}),
        ("diffusion_scaling", {"mode": "var", "clip_eps": 1e-4}),
    ]
)

register_wrapper_config(
    name="LayerNormMLPScoreModelScaled",
    base_model="LayerNormMLPScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-6})]
)

# ------------------------------------------------------------------
# UNet1D Family (TrajectorySpace)
# ------------------------------------------------------------------

register_wrapper_config(
    name="UNet1DTime",
    base_model="UNet1DTimeScoreModel",
    wrappers=[("diffusion_scaling", {"mode": "var", "clip_eps": 1e-4})]
)

register_wrapper_config(
    name="UNet1DTimeSmall",
    base_model="UNet1DTimeScoreModel",
    base_kwargs={"channel_mults": (1, 2, 4)},
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-4})]
)

register_wrapper_config(
    name="UNet1DTimeSmall2",
    base_model="UNet1DTimeScoreModel",
    base_kwargs={"channel_mults": (1, 4, 4)},
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-4})]
)

register_wrapper_config(
    name="UNet1DTimeV",
    base_model="UNet1DTimeScoreModel",
    wrappers=[("vparam", {"clip_eps": 1e-4})]
)

# ------------------------------------------------------------------
# TemporalUNet Family (TrajectorySpace)
# ------------------------------------------------------------------

register_wrapper_config(
    name="temporal_unet",
    base_model="TemporalUnetScoreModel",
    base_kwargs={"dim_mults": (1, 2, 4)},
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-6})]
)

register_wrapper_config(
    name="TemporalUNet",
    base_model="TemporalUnetScoreModel",
    base_kwargs={},
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-5})]
)

register_wrapper_config(
    name="TemporalUNetV",
    base_model="TemporalUnetScoreModel",
    base_kwargs={"dim_mults": (1, 2, 4)},
    wrappers=[("vparam", {})]
)

register_wrapper_config(
    name="TemporalUNet2",
    base_model="TemporalUnetScoreModel",
    base_kwargs={"dim_mults": (1, 2, 4, 4)},
    wrappers=[("diffusion_scaling", {"mode": "var", "clip_eps": 1e-4})]
)

register_wrapper_config(
    name="TemporalUNetSmall",
    base_model="TemporalUnetScoreModel",
    base_kwargs={"dim_mults": (1, 2, 2)},
    wrappers=[("diffusion_scaling", {"mode": "sqrt_var", "clip_eps": 1e-5})]
)

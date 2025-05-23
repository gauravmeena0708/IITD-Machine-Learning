from .eegnet_base import build as eegnet_base
from .eegnet_attn import build as eegnet_attn
from .eegnet_transformer import build as eegnet_transformer
from .eegnet_deepconv import build as eegnet_deepconv

from .qeegnet_base import build as qeegnet_base
from .qeegnet_attn import build as qeegnet_attn
from .qeegnet_transformer import build as qeegnet_transformer
from .qeegnet_deepconv import build as qeegnet_deepconv

from .msd_base import build as msd_base
from .msd_attn import build as msd_attn
from .msd_transformer import build as msd_transformer
from .msd_deepconv import build as msd_deepconv


# Define public interface
__all__ = [
    "eegnet_base", "eegnet_attn", "eegnet_transformer", "eegnet_deepconv",
    "qeegnet_base", "qeegnet_attn", "qeegnet_transformer", "qeegnet_deepconv",
    "msd_base", "msd_attn", "msd_transformer", "msd_deepconv",
    "get_model_by_name", "list_available_models"
]

MODEL_REGISTRY = {
    #EEGNet
    "eegnet_base": eegnet_base,
    "eegnet_attn": eegnet_attn,
    "eegnet_transformer": eegnet_transformer,
    "eegnet_deepconv": eegnet_deepconv,

    # Q-EEGNet
    "qeegnet_base": qeegnet_base,
    "qeegnet_attn": qeegnet_attn,
    "qeegnet_transformer": qeegnet_transformer,
    "qeegnet_deepconv": qeegnet_deepconv,

    # MSD
    "msd_base": msd_base,
    "msd_attn": msd_attn,
    "msd_transformer": msd_transformer,
    "msd_deepconv": msd_deepconv,

}

def get_model_by_name(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{name}'. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]

def list_available_models():
    return list(MODEL_REGISTRY.keys())

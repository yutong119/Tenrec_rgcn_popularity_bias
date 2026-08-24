"""Model registry and exports."""

from legacy.base_recommender import BaseRecommender
from .rgcn_baseline import RGCNBaseline
from legacy.rgcn_baseline_fullmap import GlobalRGCNBaseline
from .rgcn_penalty import RGCNPenalty
from legacy.rgcn_penalty_fullmap import GlobalRGCNPenalty
from .rgcn_penalty_gravity import RGCNPenaltyGravity
from .rgcn_penalty_gravity_fullmap import GlobalRGCNPenaltyGravity

__all__ = [
    "BaseRecommender",
    "RGCNBaseline",
    "GlobalRGCNBaseline",
    "RGCNPenalty",
    "GlobalRGCNPenalty",
    "RGCNPenaltyGravity",
    "GlobalRGCNPenaltyGravity",
]

MODEL_REGISTRY = {
    "rgcn_baseline": RGCNBaseline,
    "rgcn_baseline_fullmap": GlobalRGCNBaseline,
    "rgcn_penalty": RGCNPenalty,
    "rgcn_penalty_fullmap": GlobalRGCNPenalty,
    "rgcn_penalty_gravity": RGCNPenaltyGravity,
    "rgcn_penalty_gravity_fullmap": GlobalRGCNPenaltyGravity,
}


def _try_register_optional(module_name: str, class_name: str, registry_name: str):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        model_cls = getattr(module, class_name)
    except (ImportError, AttributeError):
        return
    globals()[class_name] = model_cls
    __all__.append(class_name)
    MODEL_REGISTRY[registry_name] = model_cls


_try_register_optional("mf", "MF", "mf")
_try_register_optional("ngcf", "NGCF", "ngcf")
_try_register_optional("lightgcn", "LightGCN", "lightgcn")
_try_register_optional("ablation_a", "AblationA", "ablation_a")
_try_register_optional("ablation_b", "AblationB", "ablation_b")
_try_register_optional("ablation_c", "AblationC", "ablation_c")
_try_register_optional("ablation_d", "AblationD", "ablation_d")


def build_model(model_name, **kwargs):
    """Build a model from the project registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} is not registered. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)

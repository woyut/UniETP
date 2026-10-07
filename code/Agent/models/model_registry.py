from typing import Optional

from Agent.models.general_model import GeneralModel
from Agent.models.api_model import APIModel

MODEL_REGISTRY = {
    "api": APIModel,
}


def get_model(
    model_type: str, model_name: str, model_config_file: Optional[str]
) -> GeneralModel:
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type={model_type!r}; choose from {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](model_name, model_config_file)

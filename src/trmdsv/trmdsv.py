from enum import Enum
from typing import Optional, Tuple, Union

import torch
from torchvision.transforms import Normalize, Resize

from .third_party.depthanything.dpt import DPT_DINOv2 as DepthAnything
from .third_party.midas.dpt_depth import DPTDepthModel as Midas

transforms = {
    "midas": (Resize((384, 384), antialias=True), Normalize(0.5, 0.5)),
    "depthanything": (
        Resize((378, 378), antialias=True),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ),
}


class MODEL(Enum):
    version = "model_release"
    DA_SUP_AUG = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{version}/da-sup-aug.pt"
    DA_SUP_TEMP_AUG = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{version}/da-sup-temp-aug.pt"
    DA_SUP_TEMP = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{version}/da-sup-temp.pt"
    DA_SUP = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{version}/da-sup.pt"


def load_model(
    model_type: str, weights_path: Optional[Union[str, MODEL]], device: str
) -> Tuple[torch.nn.Module, Resize, Normalize]:
    r"""Load a model with the given type and weights.

    Args:
        model_type (str): The type of model to load.
        weights_path (Optional[Union[str, MODEL]]): The path to the weights file or a MODEL enum value.
        device (str): The device to load the model on.

    Returns:
        Tuple[torch.nn.Module, Resize, Normalize]: The model, resize transform, and normalise transform.
    """

    # Load base model...
    if model_type == "midas":
        model = Midas(path=None, backbone="beitl16_512", non_negative=True)
    elif model_type == "depthanything":
        model = DepthAnything()
    else:
        raise ValueError(f'Incorrect base model name "{model_type}"')

    model.to(device)

    # Load weights if needed...
    resize, normalise = transforms[model_type]
    if isinstance(weights_path, str):
        if weights_path == "random":
            for param in model.parameters():
                torch.nn.init.uniform(param.data)
        else:
            state_dict = torch.load(weights_path, map_location=device)
            if model_type == "midas":
                keys = [k for k in state_dict if "attn.relative_position_index" in k]
                for k in keys:
                    del state_dict[k]
            model.load_state_dict(state_dict)
    if isinstance(weights_path, MODEL):
        weights_path = weights_path.value
        state_dict = torch.hub.load_state_dict_from_url(
            weights_path, map_location=device
        )

    return model, resize, normalise

from enum import Enum
from typing import Tuple, Union

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
    __tag = "model_release"
    DA_SUP_AUG = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{__tag}/da-sup-aug.pt"
    DA_SUP_TEMP_AUG = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{__tag}/da-sup-temp-aug.pt"
    DA_SUP_TEMP = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{__tag}/da-sup-temp.pt"
    DA_SUP = f"https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/download/{__tag}/da-sup.pt"


def load_model(
    model_type: str,
    weights_path: Union[str, MODEL] = MODEL.DA_SUP_TEMP,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, Resize, Normalize]:
    r"""Load a model with the given type and weights.

    Args:
        model_type (str): The type of model to load.
        weights_path (Union[str, MODEL]): The path to the weights file or a MODEL enum value. Defaults to MODEL.DA_SUP_TEMP. Use weight_path="random" to initialise the model with random weights.
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
    state_dict = None
    if isinstance(weights_path, str):
        if weights_path == "random":
            for param in model.parameters():
                torch.nn.init.uniform(param.data)
        else:
            state_dict = torch.load(weights_path, map_location=device)
    if isinstance(weights_path, MODEL):
        state_dict = torch.hub.load_state_dict_from_url(
            weights_path.value, map_location=device
        )

    if state_dict is not None:
        if model_type == "depthanything" and any(
            key.startswith("scratch") for key in state_dict
        ):
            raise ValueError("Provided weights seem to be for the MiDaS architecture")
        elif model_type == "midas" and any(
            key.startswith("depth_head") for key in state_dict
        ):
            raise ValueError(
                "Provided weights seem to be for the DepthAnything architecture"
            )
        if model_type == "midas":
            keys = [k for k in state_dict if "attn.relative_position_index" in k]
            for k in keys:
                del state_dict[k]
        model.load_state_dict(state_dict)

    resize, normalise = transforms[model_type]

    return model, resize, normalise

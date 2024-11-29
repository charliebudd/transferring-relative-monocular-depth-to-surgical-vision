import torch
from torchvision.transforms import Resize, Normalize
from .third_party.depthanything.dpt import DepthAnything
from .third_party.midas.dpt_depth import DPTDepthModel as Midas

transforms = {
    "midas": (Resize((384, 384), antialias=True), Normalize(0.5, 0.5)),
    "depthanything": (Resize((378, 378), antialias=True), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])),
}

def load_model(model_type, weights_path, device):
    
    # Load base model...
    if model_type == "midas":
        model = Midas(path=None, backbone="beitl16_512", non_negative=True)
    elif model_type == "depthanything":
        model = DepthAnything().from_pretrained("LiheYoung/depth_anything_vitl14")
    else:
        raise ValueError(f"Incorrect base model name \"{model_type}\"")

    model.to(device)

    # Load weights if needed...
    if weights_path != None:
        state_dict = torch.load(weights_path, map_location=device)
        
        if model_type == "depthanything" and any(key.startswith('scratch') for key in state_dict):
            raise ValueError("Provided weights seem to be for the MiDaS architecture")
        elif model_type == "midas" and any(key.startswith('depth_head') for key in state_dict):
            raise ValueError("Provided weights seem to be for the DepthAnything architecture")
        
        if model_type == "midas":
            keys = [k for k in state_dict if "attn.relative_position_index" in k]
            for k in keys:
                del state_dict[k]
        model.load_state_dict(state_dict)

    resize, normalise = transforms[model_type]

    return model, resize, normalise

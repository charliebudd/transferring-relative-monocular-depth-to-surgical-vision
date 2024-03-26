import torch
from torchvision.transforms import Resize, Normalize
from .third_party.depthanything.dpt import DPT_DINOv2 as DepthAnything
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
        model = DepthAnything()
    else:
        raise ValueError(f"Incorrect base model name \"{model_type}\"")

    model.to(device)

    # Load weights if needed...
    if weights_path != None:
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

    resize, normalise = transforms[model_type]

    return model, resize, normalise

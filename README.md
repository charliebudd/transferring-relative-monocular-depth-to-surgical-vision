# Transferring Relative Monocular Depth to Surgical Vision with Temporal Consistency
This is the official repository for our state-of-the-art approach to monocular depth in surgical vision as presented in our paper...
<ul><b>Transferring Relative Monocular Depth to Surgical Vision with Temporal Consistency</b><br>
    Charlie Budd, Tom Vercauteren.<br>
    [ <a href="https://arxiv.org/abs/2403.06683">arXiv</a> ] 
</ul>

# Using Our Models
First, install our package...
```
pip install git+https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision
```
The model may then be used as follows (`WEIGHTS_URL.DEPTHANYTHING_SUP_TEMP` best model):
```python
import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize

from trmdsv import WEIGHTS_URL, load_model

model, resize_for_model, normalise_for_model = load_model(
    model_type="depthanything",
    weights_path=WEIGHTS_URL.DEPTHANYTHING_SUP_TEMP_AUG,
    device="cuda",
)
model.eval()

image = read_image("data/cholec80_sample.png").cuda() / 255.0
original_size = image.shape[-2:]
image_for_model = normalise_for_model(resize_for_model(image.unsqueeze(0)))

with torch.no_grad():
    depth = model(image_for_model)

depth = resize(depth, original_size)

plt.subplot(121).axis("off")
plt.imshow(image.cpu().permute(1, 2, 0))
plt.subplot(122).axis("off")
plt.imshow(depth.cpu().permute(1, 2, 0))
plt.show()
```

# Recreating Our Results
\### awaiting publication \###

# Transferring Relative Monocular Depth to Surgical Vision with Temporal Consistency (MICCAI 2024)

![Example monocular depth inference](assets/example.png)

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
Then download one of our models weights from the [release tab](https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision/releases/tag/model_release) in this repo. We would recommend our best performer, `depthanything-sup-temp.pt`. The model may then be used as follows...
```python
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt

from trmdsv import load_model

model, resize_for_model, normalise_for_model = load_model("depthanything", "weights/path.pt", "cuda")
model.eval()

image = read_image("surgical_image.png").cuda() / 255.0
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

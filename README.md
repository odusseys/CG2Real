### Prerequisites

- CUDA (tested on cu121 only for now)
- pytorch 2.0+

Install additional dependencies:

```
pip install -r requirements.txt
```

### Usage

```
from cg2real import CG2Real
from PIL import Image

cg = CG2Real()

image = Image.open("your image path here")
realistic = cg("prompt to describe your image", image)
```

If using less than 16GB-ish VRAM, you can call `cg = CG2Real(low_memory=True)` to offload pipeline components as they are called ; this will slow down inference.

To increase or decrease fidelity to the original image, use: `realistic = cg("prompt to describe your image", image, fidelity=[value between 0 and 1])`

To run more denoising iterations, use the `iterations` parameter.

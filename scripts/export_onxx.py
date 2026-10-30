import sys
from pathlib import Path

import torch
from dependencies.vggt.vggt.models.vggt import VGGT 

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 1. Load model
model = VGGT.from_pretrained("facebook/VGGT-1B").eval()

# 2. Prepare dummy input (Batch size=1, Frames=2, Channels=3, H=224, W=224)
dummy_input = torch.randn(1, 2, 3, 224, 224)

# 3. Export to ONNX (Opset 17+ recommended for vision transformers)
torch.onnx.export(
    model,
    dummy_input,
    "vggt_fixed.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input_images'],
    output_names=['cameras', 'depth_maps', 'point_maps']
)
print("Exported vggt_fixed.onnx successfully!")
import os
import sys
from pathlib import Path

# 1. Resolve repository root and add VGGT dependency folder to sys.path FIRST
ROOT_DIR = Path(__file__).resolve().parent.parent
VGGT_PATH = ROOT_DIR / "dependencies" / "vggt"

if str(VGGT_PATH) not in sys.path:
    sys.path.insert(0, str(VGGT_PATH))

import torch
import torch.nn.functional as F

# 2. Patch interpolate to disable anti-aliasing (unsupported by ONNX Opset 17)
_original_interpolate = F.interpolate

def _patched_interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
    if antialias:
        antialias = False
    return _original_interpolate(
        input, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias
    )

F.interpolate = _patched_interpolate

# 3. Now safe to import VGGT
from vggt.models.vggt import VGGT 

output_dir = "./exported_models"
onnx_filename = "vggt_model.onnx"
output_path = os.path.join(output_dir, onnx_filename)

os.makedirs(output_dir, exist_ok=True)

# 4. Load model in evaluation mode
model = VGGT.from_pretrained("facebook/VGGT-1B").eval()

# 5. Prepare static dummy input (Batch size=1, Frames=2, Channels=3, H=224, W=224)
dummy_input = torch.randn(1, 2, 3, 224, 224)

# 6. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input_images'],
    output_names=['cameras', 'depth_maps', 'point_maps']
)

print(f"Exported {onnx_filename} successfully to {os.path.abspath(output_path)}!")
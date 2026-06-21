"""
export_onnx.py

Export the best saved model to ONNX format so it can be deployed
outside of PyTorch (e.g. OpenCV, ONNX Runtime, TensorRT).
"""

import torch
import torch.onnx
from model import build_model

model = build_model()
# load best weights and move to CPU for export
model.load_state_dict(torch.load("best.pth", map_location="cpu"))
model.to("cpu")
model.eval()

# dummy input to trace the model graph: 1 image, 3 channels (RGB), 256x256
dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,       # save trained weights inside the onnx file
    opset_version=11,         # onnx opset version (11 is widely supported)
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        # batch_size dimension is dynamic so we can run inference on any batch size
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("model exported to model.onnx")
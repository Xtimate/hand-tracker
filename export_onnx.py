import numpy as np
import onnxruntime as ort
import torch

from model import HandNet

ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
size = ckpt["size"]
model = HandNet(size=size)
model.load_state_dict(ckpt["model"])
model.eval()

torch.onnx.export(
    model, torch.randn(1, 3, size, size), "hand.onnx",
    input_names=["image"], output_names=["simcc_x", "simcc_y"],
    dynamic_axes={"image": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}},
    opset_version=17, dynamo=False,
)

sess = ort.InferenceSession("hand.onnx", providers=["CPUExecutionProvider"])
x = torch.randn(2, 3, size, size)
with torch.no_grad():
    tx, ty = model(x)
ox, oy = sess.run(None, {"image": x.numpy()})
print("max abs diff:", np.abs(tx.numpy() - ox).max(), np.abs(ty.numpy() - oy).max())

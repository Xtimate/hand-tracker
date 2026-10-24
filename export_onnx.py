import sys

import numpy as np
import onnxruntime as ort
import torch

from model import HandNet

ckpt = torch.load(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt", map_location="cpu")
out = sys.argv[2] if len(sys.argv) > 2 else "hand.onnx"
size = ckpt["size"]
model = HandNet(size=size)
model.load_state_dict(ckpt["model"])
model.eval()

torch.onnx.export(
    model, torch.randn(1, 3, size, size), out, # type: ignore
    input_names=["image"], output_names=["simcc_x", "simcc_y", "presence"],
    dynamic_axes={"image": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}, "presence": {0: "batch"}},
    opset_version=17, dynamo=False,
)

sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
x = torch.randn(2, 3, size, size)
with torch.no_grad():
    tx, ty, tp = model(x)
ox, oy, op = sess.run(None, {"image": x.numpy()})
print("max abs diff:", max(np.abs(tx.numpy() - ox).max(), np.abs(ty.numpy() - oy).max(), np.abs(tp.numpy() - op).max()))

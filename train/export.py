import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np
from eval import Net

net = Net(); net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model.pth"), map_location="cpu")); net.eval()
out_path = os.path.join(os.path.dirname(__file__), "..", "model.onnx")
dummy = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    net, dummy, out_path,
    input_names=["input"], output_names=["heatmap"],
    dynamic_axes={"input": {2: "h", 3: "w"}, "heatmap": {2: "h4", 3: "w4"}},
    opset_version=12,
)
print("exported", out_path, "size", round(os.path.getsize(out_path)/1024, 1), "KB")

# verify with onnxruntime at a couple of region sizes
import onnxruntime as ort
sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
for hw in [(512, 400), (845, 1514)]:
    o = sess.run(None, {"input": np.random.randn(1, 3, hw[0], hw[1]).astype(np.float32)})
    print("input", hw, "-> heatmap", o[0].shape)
# parity check vs torch
xt = np.random.randn(1, 3, 300, 500).astype(np.float32)
with torch.no_grad():
    tt = net(torch.from_numpy(xt)).numpy()
oo = sess.run(None, {"input": xt})[0]
print("torch vs onnx max abs diff:", float(np.abs(tt - oo).max()))

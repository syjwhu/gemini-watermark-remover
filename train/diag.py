import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np, cv2
from PIL import Image
from eval import Net  # reuse

net = Net().cuda(); net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model.pth"))); net.eval()

img = np.asarray(Image.open(r"C:\Users\syj\Downloads\Gemini_Generated_Image_qefaxzqefaxzqefa.png").convert("RGB"))
H, W, _ = img.shape
rx0, ry0 = int(W*0.45), int(H*0.45)
reg = img[ry0:, rx0:].astype(np.float32) / 255.0
x = torch.from_numpy(reg.transpose(2,0,1)[None]).cuda()
with torch.no_grad():
    hm = net(x)[0,0].cpu().numpy()
hm_up = cv2.resize(hm, (reg.shape[1], reg.shape[0]))
# response at the TRUE watermark (2512,1295)
tx, ty = 2512 - rx0, 1295 - ry0
print("global max %.3f at region (%d,%d) -> image (%d,%d)" % (
    hm.max(), np.unravel_index(hm.argmax(), hm.shape)[1]*4, np.unravel_index(hm.argmax(), hm.shape)[0]*4,
    rx0 + np.unravel_index(hm.argmax(), hm.shape)[1]*4, ry0 + np.unravel_index(hm.argmax(), hm.shape)[0]*4))
print("response AT true watermark (2512,1295): %.3f" % hm_up[ty, tx])
# overlay heatmap (red) on the region
ov = (reg*255).astype(np.uint8).copy()
ov[..., 0] = np.clip(ov[..., 0].astype(float) + hm_up*255, 0, 255).astype(np.uint8)
Image.fromarray(ov).save(r"C:\Users\syj\Downloads\_heat.png")
print("saved heatmap overlay (red = model response)")

import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, numpy as np, cv2
from PIL import Image, ImageDraw

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def cbr(i, o, k=3, st=1, d=1):
            return nn.Sequential(nn.Conv2d(i, o, k, st, ((k-1)//2)*d, dilation=d),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            cbr(3,16), cbr(16,16,3,2), cbr(16,32), cbr(32,32,3,2),
            cbr(32,64), cbr(64,64,3,1,2), cbr(64,64,3,1,4), nn.Conv2d(64,1,1))
    def forward(self, x): return torch.sigmoid(self.net(x))

net = Net().cuda(); net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model.pth"))); net.eval()

def detect(img):
    H, W, _ = img.shape
    rx0, ry0 = int(W*0.45), int(H*0.45)
    reg = img[ry0:, rx0:].astype(np.float32) / 255.0
    x = torch.from_numpy(reg.transpose(2,0,1)[None]).cuda()
    with torch.no_grad():
        hm = net(x)[0,0].cpu().numpy()
    py, px = np.unravel_index(hm.argmax(), hm.shape)
    return rx0 + px*4, ry0 + py*4, float(hm.max())

files = sorted(glob.glob(r"C:\Users\syj\Downloads\Gemini_Generated_Image_*.png"))
same = [f for f in files if Image.open(f).size == (2752, 1536)]
cols = 5; rows = (len(same)+cols-1)//cols; cw, ch = 300, 200
M = Image.new("RGB", (cw*cols, ch*rows), (15,15,15)); d = ImageDraw.Draw(M)
hi = 0
for i, f in enumerate(same):
    im = np.asarray(Image.open(f).convert("RGB"))
    cx, cy, conf = detect(im)
    if conf > 0.3: hi += 1
    crop = Image.open(f).convert("RGB").crop((cx-150, cy-100, cx+150, cy+100)).resize((cw, ch))
    dd = ImageDraw.Draw(crop); dd.ellipse([150-9,100-9,150+9,100+9], outline=(0,255,0), width=3)
    x = (i%cols)*cw; y = (i//cols)*ch; M.paste(crop, (x,y))
    d.text((x+2,y+2), "%s c=%.2f" % (os.path.basename(f)[23:27], conf), fill=(255,255,0))
M.save(r"C:\Users\syj\Downloads\_mleval.png")
print("detected (conf>0.3): %d / %d images" % (hi, len(same)))

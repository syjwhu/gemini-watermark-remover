import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np, cv2
from PIL import Image, ImageDraw
from eval import Net, detect

net = Net().cuda(); net.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "model.pth"))); net.eval()
import eval as E; E.net = net

files = sorted(glob.glob(r"C:\Users\syj\Downloads\Gemini_Generated_Image_*.png"))
same = [f for f in files if Image.open(f).size == (2752, 1536)]
cols = 6; rows = (len(same)+cols-1)//cols; cw = ch = 200
M = Image.new("RGB", (cw*cols, ch*rows), (15,15,15)); d = ImageDraw.Draw(M)
for i, f in enumerate(same):
    im = np.asarray(Image.open(f).convert("RGB"))
    cx, cy, conf = detect(im)
    R = 70
    cx = max(R, min(im.shape[1]-R, cx)); cy = max(R, min(im.shape[0]-R, cy))
    crop = im[cy-R:cy+R, cx-R:cx+R].astype(np.float32)
    # enhance faint white: amplify deviation above a heavy blur
    blur = cv2.GaussianBlur(crop, (0,0), 9)
    enh = np.clip(crop + (crop-blur)*3.0, 0, 255).astype(np.uint8)
    im2 = Image.fromarray(enh).resize((cw, ch), Image.NEAREST)
    dd = ImageDraw.Draw(im2)
    dd.line([cw//2-12, ch//2, cw//2+12, ch//2], fill=(0,255,0), width=2)
    dd.line([cw//2, ch//2-12, cw//2, ch//2+12], fill=(0,255,0), width=2)
    x = (i%cols)*cw; y = (i//cols)*ch; M.paste(im2, (x,y))
    d.text((x+2,y+2), "c=%.2f" % conf, fill=(255,255,0))
M.save(r"C:\Users\syj\Downloads\_verify.png")
print("saved: crosshair = model detection, contrast-enhanced to reveal faint sparkles")

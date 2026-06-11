import numpy as np, cv2, glob, os

SIZE = 256          # training patch size
HM = SIZE // 4      # heatmap resolution (/4)
rng = np.random.default_rng(0)

# load the watermark alpha (the star shape), tight + normalized
A = np.load(os.path.join(os.path.dirname(__file__), "..", "sparkle_alpha.npy")).astype(np.float32)
ys, xs = np.where(A > 0.02); A = A[ys.min():ys.max()+1, xs.min():xs.max()+1]; A = A / A.max()

# load backgrounds (the user's busy figures)
_files = sorted(glob.glob(r"C:\Users\syj\Downloads\Gemini_Generated_Image_*.png"))
BG = []
for f in _files:
    try:
        im = cv2.imread(f)
        if im is None: continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im.shape[0] >= SIZE and im.shape[1] >= SIZE:
            BG.append(im)
    except Exception:
        pass
print("loaded %d background images" % len(BG))

# precompute a gaussian peak
def gauss(cx, cy, sigma=1.6):
    yy, xx = np.mgrid[0:HM, 0:HM]
    return np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma * sigma)).astype(np.float32)

def _distractor(crop):
    """Stamp a bright/white shape that is NOT the star (hard negative)."""
    kind = rng.integers(3); op = float(rng.uniform(0.4, 0.95))
    cx = int(rng.integers(30, SIZE-30)); cy = int(rng.integers(30, SIZE-30))
    if kind == 0:                                   # white rectangle (panel/box)
        w = int(rng.integers(20, 130)); h = int(rng.integers(10, 90))
        x0, y0 = max(0,cx-w//2), max(0,cy-h//2); x1, y1 = min(SIZE,cx+w//2), min(SIZE,cy+h//2)
        crop[y0:y1, x0:x1] = (1-op)*crop[y0:y1, x0:x1] + op*255.0
    elif kind == 1:                                 # white disc/blob
        r = int(rng.integers(12, 60)); yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        m = ((xx-cx)**2 + (yy-cy)**2) <= r*r
        crop[m] = (1-op)*crop[m] + op*255.0
    else:                                           # bright bars (text-like)
        for _ in range(int(rng.integers(2,6))):
            yb = int(rng.integers(20, SIZE-20)); xb = int(rng.integers(20, SIZE-70)); wb = int(rng.integers(15, 70))
            crop[yb:yb+3, xb:xb+wb] = (1-op)*crop[yb:yb+3, xb:xb+wb] + op*255.0

def make_batch(n, pos_ratio=0.55):
    X = np.zeros((n, 3, SIZE, SIZE), np.float32)
    Y = np.zeros((n, 1, HM, HM), np.float32)
    for i in range(n):
        bg = BG[rng.integers(len(BG))]
        y0 = rng.integers(0, bg.shape[0] - SIZE + 1); x0 = rng.integers(0, bg.shape[1] - SIZE + 1)
        crop = bg[y0:y0+SIZE, x0:x0+SIZE].astype(np.float32).copy()
        if rng.random() < 0.5:
            crop = crop[:, ::-1].copy()
        u = rng.random()
        if u < pos_ratio:                            # POSITIVE: the real star
            s = int(rng.integers(40, 181))
            a = cv2.resize(A, (s, s)) * float(rng.uniform(0.55, 1.0))
            a = a[..., None]
            cx = int(rng.integers(s//2 + 3, SIZE - s//2 - 3)); cy = int(rng.integers(s//2 + 3, SIZE - s//2 - 3))
            yy0, xx0 = cy - s//2, cx - s//2
            reg = crop[yy0:yy0+s, xx0:xx0+s]
            crop[yy0:yy0+s, xx0:xx0+s] = (1 - a) * reg + a * 255.0
            Y[i, 0] = gauss(cx / 4.0, cy / 4.0)
        elif u < pos_ratio + 0.25:                   # HARD NEGATIVE: white distractor, empty label
            for _ in range(int(rng.integers(1, 3))): _distractor(crop)
        # else: plain negative (busy figure, no watermark), empty label
        X[i] = crop.transpose(2, 0, 1) / 255.0
    return X, Y

if __name__ == "__main__":
    from PIL import Image, ImageDraw
    X, Y = make_batch(8)
    cells = []
    for i in range(8):
        img = (X[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        im = Image.fromarray(img); d = ImageDraw.Draw(im)
        hm = Y[i, 0]
        if hm.max() > 0.5:
            py, px = np.unravel_index(hm.argmax(), hm.shape)
            d.ellipse([px*4-8, py*4-8, px*4+8, py*4+8], outline=(0, 255, 0), width=2)
        else:
            d.text((6, 6), "NEG", fill=(255, 0, 0))
        cells.append(im)
    M = Image.new("RGB", (SIZE*4, SIZE*2), (0, 0, 0))
    for i, im in enumerate(cells):
        M.paste(im, ((i % 4)*SIZE, (i//4)*SIZE))
    M.save(r"C:\Users\syj\Downloads\_datagen.png")
    print("saved sample viz; green = stamped watermark + label, NEG = no watermark")

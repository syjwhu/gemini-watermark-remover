import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn
from dataset import make_batch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def cbr(i, o, k=3, st=1, d=1):
            return nn.Sequential(nn.Conv2d(i, o, k, st, ((k-1)//2)*d, dilation=d),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(
            cbr(3, 16), cbr(16, 16, 3, 2),     # 256 -> 128
            cbr(16, 32), cbr(32, 32, 3, 2),    # 128 -> 64
            cbr(32, 64), cbr(64, 64, 3, 1, 2), # 64, dilated (big receptive field)
            cbr(64, 64, 3, 1, 4),
            nn.Conv2d(64, 1, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.net(x))

dev = "cuda"
net = Net().to(dev)
print("params:", sum(p.numel() for p in net.parameters()))
opt = torch.optim.Adam(net.parameters(), 1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 5000)

def loss_fn(pred, tgt):
    w = 1.0 + 15.0 * (tgt > 0.1).float()      # weight the peak region
    return (w * (pred - tgt) ** 2).mean()

ITERS, BS = 5000, 32
t0 = time.time()
for it in range(1, ITERS + 1):
    X, Y = make_batch(BS)
    X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
    pred = net(X); loss = loss_fn(pred, Y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 250 == 0:
        print("it %4d  loss %.5f  peak~%.2f  (%.0fs)" % (it, loss.item(), pred.max().item(), time.time()-t0))

torch.save(net.state_dict(), os.path.join(os.path.dirname(__file__), "model.pth"))
print("saved model.pth")

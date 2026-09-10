"""Train the score model on CIFAR-10 by denoising score matching.

    python train.py                       -> artifacts/score_cifar.pt, artifacts/train.log
    python train.py --bench               time 40 steps and exit, to size the run

The data is downloaded to data/ on first use, scaled to [-1, 1] and flipped
horizontally at random.  For each image a time t ~ U(T_MIN, 1) and a noise
eps are drawn, x_t = alpha(t) x_0 + sigma(t) eps is formed, and the network
is trained to return eps; the score is then s_theta = -eps_theta / sigma
(sde.score).  An exponential moving average of the weights is what gets
saved and sampled from.  The checkpoint is rewritten after every epoch, so
the run can be sampled from at any time.
"""

import copy
import os
import sys
import time

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import DiT
from sde import alpha, sigma

EPOCHS = 500                 # benchmark: 2094 images/s, 23.9 s per epoch -> about 3.3 hours
BATCH = 256                  # the DiT batch; 512 needs more activation memory than the card has
LR = 2e-4                    # AdamW without weight decay, the DiT recipe
EMA = 0.9999
T_MIN = 1e-3
SEED = 0
OUT = os.path.join(HERE, "artifacts", "score_cifar.pt")


def main(bench=False):
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data = datasets.CIFAR10(os.path.join(HERE, "data"), train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(data, batch_size=BATCH, shuffle=True, drop_last=True,
                                         num_workers=4, pin_memory=True, persistent_workers=True)
    model = DiT().to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters, "
          f"{len(loader)} batches of {BATCH} per epoch, {EPOCHS} epochs", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t0 = time.time()
    step = 0
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for x0, _ in loader:
            x0 = x0.to(dev, non_blocking=True)
            t = T_MIN + (1.0 - T_MIN) * torch.rand(x0.shape[0], device=dev)
            eps = torch.randn_like(x0)
            xt = alpha(t)[:, None, None, None] * x0 + sigma(t)[:, None, None, None] * eps
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.mse_loss(model(xt, t).float(), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(EMA).add_(pm, alpha=1.0 - EMA)
            total += loss.item()
            step += 1
            print(f"    epoch {epoch} step {step}: loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if bench and step == 50:
                torch.cuda.synchronize()
                t1 = time.time()
            if bench and step == 90:
                torch.cuda.synchronize()
                rate = 40 * BATCH / (time.time() - t1)
                print(f"benchmark: {rate:.0f} images/s, {len(data)/rate:.1f}s per epoch, "
                      f"{2.5*3600*rate/len(data):.0f} epochs in 2.5 h", flush=True)
                return
        torch.save({"ema": ema.state_dict(), "epoch": epoch, "step": step,
                    "t_min": T_MIN}, OUT)
        print(f"epoch {epoch}/{EPOCHS}: mean loss {total/len(loader):.4f}, "
              f"{time.time()-t0:.0f}s, saved {OUT}", flush=True)


if __name__ == "__main__":
    main(bench="--bench" in sys.argv)

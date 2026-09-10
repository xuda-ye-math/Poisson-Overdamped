"""Sample CIFAR-10 images with the four integrators, one shared Brownian path each.

    python sample.py [N] [n] [--keep 1,3,5]
                                N steps (60 by default), n images (6 by default);
                                --keep stores only the listed paths (1-based) of the
                                seeded run, with their numbers -> artifacts/samples_N{N}.npz

For each of N_SAMPLES images one Brownian path is drawn on a fine grid of
N_REF steps over the reverse Langevin time [0, TAU_END], as the exact pair
(dB, dA) of increment and time integral.  The reference SRK-LD chain runs on
that fine grid; the four integrators run on the same path aggregated exactly
onto N steps.  All start from the same X_T ~ N(0, I) and all end at t = T_END,
where the same Tweedie step denoises them.  What is saved is the denoised
image of every scheme and of the reference, the raw endpoints, and the root
mean square distances between them, per image.
"""

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from integrators import METHODS, coarsen, fine_noise, integrate, srk_ld
from model import DiT
from sde import T_END, TAU_END, grad_U, tweedie

N_SAMPLES = 6
N_REF = 1920                  # fine grid; 60 and 120 both divide it
SEED = 20260907
CKPT = os.path.join(HERE, "artifacts", "score_cifar.pt")


def rms(a, b):
    return torch.sqrt(((a - b) ** 2).flatten(1).mean(dim=1)).cpu().numpy()


def main(N, N_SAMPLES=N_SAMPLES, keep=None):
    assert N_REF % N == 0, (N_REF, N)
    dev = torch.device("cuda")
    ck = torch.load(CKPT, map_location=dev)
    model = DiT().to(dev).eval()
    model.load_state_dict(ck["ema"])
    gU = grad_U(model)

    shape = (N_SAMPLES, 3, 32, 32)
    gen = torch.Generator(device=dev).manual_seed(SEED)
    x_T = torch.randn(shape, generator=gen, device=dev)
    dt = TAU_END / N_REF
    dB, dA = fine_noise(gen, N_REF, shape, dt, dev)
    gen_flags = torch.Generator(device=dev).manual_seed(SEED + 1)
    flags = torch.rand((N, N_SAMPLES, 1, 1, 1), generator=gen_flags, device=dev) < 0.5

    out = {"N": N, "N_ref": N_REF, "h": TAU_END / N, "tau_end": TAU_END, "t_end": T_END,
           "seed": SEED, "epoch": int(ck["epoch"]), "methods": np.array(list(METHODS))}
    with torch.no_grad():
        t0 = time.time()
        z_ref = integrate(srk_ld, gU, x_T, dt, dB, dA)
        x_ref = tweedie(model, z_ref, T_END)
        print(f"reference: {N_REF} SRK-LD steps in {time.time()-t0:.1f}s", flush=True)
        out["raw_ref"], out["img_ref"] = z_ref.cpu().numpy(), x_ref.cpu().numpy()

        DB, DA = coarsen(dB, dA, N, dt)
        h = TAU_END / N
        raw, img = {}, {}
        for name, (step, evals, label) in METHODS.items():
            t0 = time.time()
            fl = flags if name == "random_splitting_rk3" else None
            raw[name] = integrate(step, gU, x_T, h, DB, DA, fl)
            img[name] = tweedie(model, raw[name], T_END)
            out[f"raw|{name}"], out[f"img|{name}"] = raw[name].cpu().numpy(), img[name].cpu().numpy()
            out[f"rms_ref|{name}"] = rms(img[name], x_ref)
            out[f"rms_ref_raw|{name}"] = rms(raw[name], z_ref)
            print(f"{label:28s} {N} steps, {evals} evaluations each, {time.time()-t0:.1f}s; "
                  f"rms to reference per image: "
                  + " ".join(f"{v:.4f}" for v in out[f"rms_ref|{name}"]), flush=True)
        names = list(METHODS)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out[f"rms|{a}|{b}"] = rms(img[a], img[b])

    out["paths"] = np.arange(1, N_SAMPLES + 1)
    if keep:                                          # keep only the listed paths of the seeded run
        sel = [k - 1 for k in keep]
        for key in list(out):
            if key in ("img_ref", "raw_ref", "paths") or "|" in key:
                out[key] = out[key][sel]
        print(f"kept paths {keep}")
    path = os.path.join(HERE, "artifacts", f"samples_N{N}.npz")
    np.savez_compressed(path, **out)
    print(f"wrote {path}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    keep = [int(x) for x in argv[argv.index("--keep") + 1].split(",")] if "--keep" in argv else None
    pos = [a for a in argv if not a.startswith("--") and (argv.index(a) == 0 or argv[argv.index(a) - 1] != "--keep")]
    main(int(pos[0]) if pos else 60, int(pos[1]) if len(pos) > 1 else N_SAMPLES, keep)

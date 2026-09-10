# Integrators on a score-based diffusion model: CIFAR-10

The pipeline first built for MNIST, on CIFAR-10. `sde.py` and
`integrators.py` are the same files; what changes is the data (32 x 32 x 3,
50,000 images, horizontal flips), the model, and the training budget.

## The model

`model.py` is DiT-S/2: 2 x 2 patches, 256 tokens of dimension 384, twelve
blocks of six heads with adaLN-Zero conditioning on a sinusoidal embedding of
the time, a zero-initialised final layer; 33 M parameters. It predicts the
noise `eps` in `x_t = alpha(t) x_0 + sigma(t) eps` of the variance preserving
diffusion, `beta` from 0.1 to 20 on `[0, 1]`, and the score is
`s = -eps / sigma(t)`.

`train.py` is standard denoising score matching, independent of the
integrators: uniform `t` in `[1e-3, 1]`, mean square error on `eps`, AdamW at
`2e-4`, batch 256 in bfloat16, weights averaged with decay 0.9999, the
checkpoint rewritten after every epoch. The log prints the loss at every
step. `python train.py --bench` times forty steps and prints the images per
second and the epochs that fit in 2.5 hours; `EPOCHS` is set from that: on
an RTX 5090, 2094 images/s, 23.9 s per epoch; 500 epochs (97,500 steps)
take about 3.3 hours.

```sh
python train.py --bench  # throughput
python train.py          # -> data/cifar-10-batches-py (downloaded once), artifacts/score_cifar.pt, artifacts/train.log
```

## Sampling

Identical to the MNIST study: the reverse SDE in the Langevin form
`dX = -grad U(X, tau) d tau + sqrt 2 dB`, the stop at `t = T_END = 0.12` with
a Tweedie step, one Brownian path per image drawn as the exact `(dB, dA)`
pair on a fine grid of 1920 steps and aggregated exactly onto the `N` coarse
steps, the four integrators on that path and the SRK-LD reference on the
fine grid.

```sh
python sample.py 60 12                              # N = 60 steps, 12 images -> artifacts/samples_N60.npz
python sample.py 30 12 --keep 1,3,5,6,7,8,9,12      # N = 30, the same 12 seeded paths, 8 of them stored -> artifacts/samples_N30.npz
python plot_samples.py 60 --paths 1,3,5,6,7,8,9,12  # -> results/samples_N60.png (9 x 8)
python plot_samples.py 30                           # -> results/samples_N30.png (9 x 8)
```

Everything is seeded (`SEED` in `sample.py` fixes the starting noise, the
paths and the splitting flags), so the tables are reproduced exactly by these
commands from the saved checkpoint. `--keep` stores only the listed paths of
the seeded run, with their numbers; `--paths` on the plot script selects
among the stored paths by those numbers. The tables show no path numbers;
each table has nine rows, top to bottom: the reference; stochastic Heun,
random splitting LMC (RK3), SRK-I, SRK-II; then the four differences to the
reference in the same order, `|x - x_ref|` averaged over the colour channels
on one colour scale, the three blocks separated by a small gap with a dashed grey line. The
N = 60 table shows the same eight paths as the N = 30 table, selected with
`--paths` from the twelve stored.

The plot script also prints the root mean square distances to the reference
and between the schemes, per image.

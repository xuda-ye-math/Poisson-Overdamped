# Numerical experiments

The shared model and schemes, and the studies built on them. Pure JAX on CPU.
Interpreter: `/home/xuda/.envs/jflows/bin/python`. Every script is run from
this folder and resolves its own paths from its file location, so

```sh
python finite-time/plot_strong.py
```

works whatever the working directory holds.

## Layout

```
potential.py     the 2-D two-well potential, its gradient and Hessian, X0, T_FINAL
noise.py         the exact Brownian pair (dB, dA) and its aggregation to a step size
integrators.py   the seven compared schemes, plus srk_ld, the strong-order-3/2 reference
observables.py   the family of 1-Lipschitz test functions behind both weak errors
plot_style.py    shared rcParams, colours, labels, log axes, slope guides
finite-time/     strong and weak error against the exact solution over [0, T]
long-time/       the empirical time average over a long horizon
```

The five modules above are common to every study: the model, the driving
noise, the schemes, the test functions and the figure style. A study folder
holds what is specific to it, and imports the common modules from the parent
directory.

## Common modules

| file | role |
| --- | --- |
| `potential.py` | a quadratic bowl with three Gaussian bumps carved into it, leaving two wells; nonconvex on a bounded region, every derivative bounded, so it satisfies the assumption of the manuscript |
| `noise.py` | the exact joint law of the increment `dB` and its time integral `dA` on a fine grid, and their exact aggregation onto any coarser grid that divides it |
| `integrators.py` | the schemes of Table 1, each driven by the same Brownian path, with the gradient evaluations per step that the equal-cost comparison holds fixed |
| `observables.py` | the 1-Lipschitz test functions; both studies measure a weak error over this same family, one against the exact solution and one against the invariant measure |
| `plot_style.py` | one colour and marker per scheme, the log cost axis, and the dashed slope guides |

Everything in `noise.py` is jitted, and so is the reference integrator's
block form in `integrators.py`. The arrays on the fine grid are the largest
in the study, and keeping them inside a compiled region is what lets the
sixteen workers of `finite-time/run_all.py` run at once.

## Studies

`finite-time/` — the two panels of Figure 1: strong error
$\max_k (\mathbb E |Z_k - X_{kh}|^2)^{1/2}$ and weak error
$\max_i \max_k |\mathbb E f_i(Z_k) - \mathbb E f_i(X_{kh})|$, both against the
number of evaluations of $\nabla U$. See `finite-time/README.md`.

`long-time/` — the weak error of the time average,
$\max_i |\overline{f_i} - \pi(f_i)|$, to $T = 1.6 \cdot 10^9$ from the origin,
for integrators I and II at five costs, $C = 6, 12, 24, 48, 96$ evaluations of
$\nabla U$ per unit time. Ten independent chains, 248 billion steps, sampled
once per unit time and streamed to disk in resumable segments. See
`long-time/README.md`.

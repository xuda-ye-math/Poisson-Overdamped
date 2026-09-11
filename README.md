# Mean square error analysis of stochastic Runge--Kutta integrators

Manuscript and source code for the mean square error analysis of two stochastic
Runge--Kutta integrators for overdamped Langevin dynamics with a potential that
is convex outside a bounded region. The
analysis goes through the discrete Poisson equation; the experiments measure
the finite-time strong and weak errors, the step-size bias of long time
averages, and the quality of samples from a diffusion model of CIFAR-10.

## The two integrators

The dynamics is the overdamped Langevin equation

$$
\mathrm{d}X_t = -\nabla U(X_t)\,\mathrm{d}t + \sqrt{2}\,\mathrm{d}B_t ,
$$

whose invariant measure is $\pi(x) \propto e^{-U(x)}$. A chain $(Z_k)_{k \ge 0}$
with step size $h$ approximates $X_{kh}$. Over the step $[kh, (k+1)h]$ both
integrators use the increment and the time integral of the Brownian path,

$$
\Delta B_k = B_{(k+1)h} - B_{kh}, \qquad
I_k = \int_{kh}^{(k+1)h} \bigl( B_s - B_{kh} \bigr)\,\mathrm{d}s ,
$$

a Gaussian pair that is drawn exactly. The two stochastic Runge--Kutta (SRK)
integrators, proposed by Yang & Wang (2026), evaluate $\nabla U$ and no higher
derivative.

**SRK-I**, two evaluations of $\nabla U$ per step:

$$
\begin{aligned}
H_k &= Z_k - \frac{3h}{4}\,\nabla U(Z_k) + \frac{3\sqrt{2}}{2h}\, I_k , \\
Z_{k+1} &= Z_k - \frac{h}{3}\,\nabla U(Z_k) - \frac{2h}{3}\,\nabla U(H_k)
          + \sqrt{2}\,\Delta B_k .
\end{aligned}
$$

**SRK-II**, three evaluations of $\nabla U$ per step:

$$
\begin{aligned}
H_k^{\pm} &= Z_k - \frac{h}{2}\,\nabla U(Z_k)
             + \frac{\sqrt{2}}{h}\Bigl( 1 \pm \frac{1}{\sqrt{2}} \Bigr) I_k , \\
Z_{k+1} &= Z_k - \frac{h}{2}\bigl( \nabla U(H_k^{+}) + \nabla U(H_k^{-}) \bigr)
          + \sqrt{2}\,\Delta B_k .
\end{aligned}
$$

Both are of strong order $3/2$ and weak order $2$ (Corollaries 1 and 2 of the
paper).

## Main results

The potential $U \in C^5(\mathbb{R}^d)$ has $|\nabla U(0)| \le M$ and
derivatives of orders $2$ to $5$ bounded by $M$, and is convex outside a
bounded region: $\nabla^2 U(x) \succeq m I_d$ for $|x| \ge R$. No global
convexity is assumed, so $U$ may have several wells. The test function
$f \in C^3(\mathbb{R}^d)$ has its first three derivatives bounded by $L$.

**Mean square error of the time average** (Theorem 2). For every step size
$h \le h_0$ and every number of steps $N$ with $Nh \ge 1$,

$$
\mathrm{MSE}(N,h)
= \mathbb{E}\,\biggl| \frac{1}{N} \sum_{k=0}^{N-1} f(Z_k) - \pi(f) \biggr|^2
\le C L^2 \bigl( 1 + |Z_0| \bigr)^{12} \Bigl( \frac{1}{Nh} + h^4 \Bigr) ,
$$

with $C$ and $h_0$ depending only on $m$, $M$, $R$ and $d$. Both terms are
optimal: $1/(Nh)$ is the sampling variance of a time average over a trajectory
of length $Nh$, and $h^4$ is the square of a second order bias. The bound holds
at every finite $N$, not in an ergodic limit.

**Uniform-in-time Wasserstein bound** (Theorem 3). For every $N \ge 1$,

$$
\mathcal{W}_1\bigl( \mathrm{Law}(Z_N), \mathrm{Law}(X_{Nh}) \bigr)
\le C \bigl( 1 + |Z_0| \bigr)^{6}\, h^2 \Bigl( 1 + \log \frac{1}{h} \Bigr) ,
$$

with the right side independent of $N$.

Both results rest on a discrete Poisson equation and on elliptic estimates for
its solution under convexity outside a bounded region (Theorem 1): the
Kolmogorov solution and its first three derivatives decay exponentially in
time, which is what the summation over the time steps requires.

```
Paper/
  main.tex                    the manuscript
  references.bib              the cited works
  plainnat-ima.bst            bibliography style, author-year, small-caps
                              author names
  ima-authoring-template.cls  document class (Oxford University Press, LaTeX
                              Project Public License), with the journal header,
                              history line and copyright footer switched off
  main.pdf                    the compiled manuscript
  figures/                    the figures of Section 5, copies of the results below
Codes/
  Convergence/       Sections 5.1 and 5.2: the potential, the schemes, the
                     finite-time strong and weak errors, the long-time
                     sampling error; pure JAX on CPU
  Diffusion_CIFAR/   Section 5.3: the score model, its training, and sampling
                     with the compared integrators
Formalization/
  Wick/              exact symbolic check (sympy) of the coefficients of
                     Lemmas 1 and 2, with the saved output of verify.py
  Lean/              Lean 4 formalization of the constants of Theorem 1 and
                     Lemmas 1 and 2
```

## Building the manuscript

```sh
cd Paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The class and the bibliography style are the two local files above; every
other package is on CTAN.

## Reproducing the experiments

Each code folder has its own `README.md` with the layout, the commands and the
run times:

- `Codes/Convergence/README.md` for the strong and weak errors of Figure 1 and
  the sampling error of Figure 2. The plotting scripts read saved data, so the
  figures rebuild without simulating; regenerating the data is described there.
- `Codes/Diffusion_CIFAR/README.md` for training the score model and drawing
  the samples of Figures 3 and 4.

The figures in `Paper/figures/` are copies of the files the scripts write to
`Codes/Convergence/finite-time/results/`, `Codes/Convergence/long-time/results/`
and `Codes/Diffusion_CIFAR/results/`.

## Checking the proofs

`Formalization/` holds two independent machine checks of the local error
decompositions, Lemmas 1 and 2 of the paper, whose coefficients are the part of
the proofs hardest to verify by hand.

- `Formalization/Wick/` computes the expectations of the mean-zero terms
  exactly with sympy, from the covariance of Brownian motion alone, and shows
  that every printed coefficient is the unique value that makes them vanish;
  `remainder.py` checks the decompositions as pathwise identities. The saved
  runs are `verify_output.txt` and `remainder_output.txt`.
- `Formalization/Lean/` proves the same constants, and those of Theorem 1, in
  Lean 4 with Mathlib, with the stochastic inputs as hypotheses; `lake build`
  runs with no `sorry`, and `build_output.txt` is the saved build with the
  axioms of every Theorem 1 statement.

`Formalization/README.md` describes both routes, how to run them, and what they
do not cover.

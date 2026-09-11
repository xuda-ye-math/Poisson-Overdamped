# Mean square error analysis of stochastic Runge--Kutta integrators

Manuscript and source code for the mean square error analysis of two stochastic
Runge--Kutta integrators for overdamped Langevin dynamics with a potential that
is convex outside a bounded region. The manuscript is the version submitted to
the IMA Journal of Numerical Analysis, typeset with the journal's class. The
analysis goes through the discrete Poisson equation; the experiments measure
the finite-time strong and weak errors, the step-size bias of long time
averages, and the quality of samples from a diffusion model of CIFAR-10.

```
Paper/
  main.tex                    the manuscript
  references.bib              the cited works
  plainnat-ima.bst            bibliography style: plainnat adapted to the
                              reference format of the journal
  ima-authoring-template.cls  the journal's document class (Oxford University
                              Press, LaTeX Project Public License)
  main.pdf                    the compiled manuscript
  figures/                    the figures of Section 5, copies of the results below
Codes/
  Convergence/       Sections 5.1 and 5.2: the potential, the schemes, the
                     finite-time strong and weak errors, the long-time
                     sampling error; pure JAX on CPU
  Diffusion_CIFAR/   Section 5.3: the score model, its training, and sampling
                     with the compared integrators
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

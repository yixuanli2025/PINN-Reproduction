# PINN-Reproduction

Please be aware that AI(more specifically Opus 5) is involved in both debugging and writing the readme files. 
This repository serves as a documentation of my own learning progress, and potentially for application uses

--------------------------------------------------------------------------------------------------------------------------------------------

Reproductions of two foundational physics-informed neural network papers,
implemented from scratch in PyTorch.

Built as a learning project: starting from a hand-written scalar
reverse-mode autodiff engine, through PyTorch fundamentals, to reproducing
published results. The progression is deliberate — each problem validates
something the next one assumes.

## Results

| Experiment | Reference | This repo |
|---|---|---|
| Burgers, forward (Raissi et al. 2019) | 6.7e-4 | **6.9e-4** best of 10 seeds, 2.5e-3 mean |
| Convection $\beta=30$, vanilla (Krishnapriyan et al. 2021) | fails | 8.7e-1 |
| Convection $\beta=30$, curriculum | recovers | **1.7e-2** (51× over vanilla) |

## Contents

### [01 — Heat equation](01-heat-equation/)


### [02 — Convection](02-convection/)


### [03 — Burgers](03-burgers/)

## Papers

- Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed
  neural networks: A deep learning framework for solving forward and inverse
  problems involving nonlinear partial differential equations.* Journal of
  Computational Physics 378, 686–707.
  [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)
- Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R. & Mahoney, M. W.
  (2021). *Characterizing possible failure modes in physics-informed neural
  networks.* NeurIPS. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)

Background reading: Karniadakis, G. E. et al. (2021). *Physics-informed
machine learning.* Nature Reviews Physics 3, 422–440.

## Setup

The Burgers reproduction requires `burgers_shock.mat` from
[maziarraissi/PINNs](https://github.com/maziarraissi/PINNs), under
`appendix/Data/`.

All experiments single run on CPU in minutes to tens of minutes.

# Burgers' Equation — reproducing Raissi et al. (2019)

Reproduction of Section 2.1 (continuous-time inference, Burgers' equation)
from *Physics Informed Deep Learning (Part I): Data-driven Solutions of
Nonlinear Partial Differential Equations*, Raissi, Perdikaris &
Karniadakis — published as *Journal of Computational Physics* 378 (2019),
686–707.

## Problem

$$u_t + u u_x - \frac{0.01}{\pi} u_{xx} = 0, \qquad x \in [-1, 1],\ t \in [0, 1]$$

- **Initial condition:** $u(0,x) = -\sin(\pi x)$
- **Boundary conditions:** $u(t,-1) = u(t,1) = 0$ (Dirichlet)

**Residual:**

$$f := u_t + u u_x - \frac{0.01}{\pi} u_{xx}$$

### The physics

Three terms in competition:

- $u_t$ — rate of change at a point
- $uu_x$ — **nonlinear self-advection**. The field transports itself, so
  faster regions move forward faster than slower ones and the profile
  steepens.
- $\frac{0.01}{\pi}u_{xx}$ — **viscous diffusion**, smoothing gradients out

Burgers is a genuine simplification of Navier–Stokes: the same nonlinear
advection plus diffusion structure, in one dimension, without pressure or
incompressibility.

## Loss

$$\mathcal{L} = \underbrace{\frac{1}{N_u}\sum_i \big(\hat u(t_u^i, x_u^i) - u^i\big)^2}_{\text{initial + boundary}} + \underbrace{\frac{1}{N_f}\sum_i \big|f(t_f^i, x_f^i)\big|^2}_{\text{PDE residual}}$$

Raissi combines the initial and boundary conditions into a single
$\mathrm{MSE}_u$ term, since both carry known target values — unlike the
periodic case in the convection section, where the boundary term is
unsupervised.

## Changes from the convection setup

| | Convection | Burgers |
|---|---|---|
| Residual | $u_t + \beta u_x$ | $u_t + uu_x - \nu u_{xx}$ |
| Linearity | linear | **nonlinear** ($uu_x$) |
| Domain in $x$ | $[0, 2\pi]$ | $[-1, 1]$ |
| Initial condition | $\sin(x)$ | $-\sin(\pi x)$ |
| Boundary | periodic (unsupervised) | Dirichlet, $u=0$ (supervised) |
| Derivatives | $u_t$, $u_x$ | $u_t$, $u_x$, $u_{xx}$ |
| Ground truth | analytic, $\sin(x - \beta t)$ | reference solution (`burgers_shock.mat`) |

## Setup (matching Raissi)

| | |
|---|---|
| Architecture | 8 hidden layers × 20 neurons, tanh — `[2, 20×8, 1]` |
| Points | $N_u = 100$ (IC + BC combined), $N_f = 10{,}000$ |
| Sampling | Latin Hypercube (`scipy.stats.qmc`) |
| Initialisation | Xavier normal, zero biases |
| Optimiser | L-BFGS only, `max_iter` 50,000, `strong_wolfe` line search |
| Evaluation | full $256 \times 100$ reference grid, relative $L^2$ |
| Target | relative $L^2$ error $= 6.7\times10^{-4}$ |

### Implementation details taken from the original code

Several choices in `Burgers.py` are not stated in the paper text but matter
for matching the result:

**Input normalisation.** Before the first layer, both inputs are rescaled
to $[-1,1]$:

```python
X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
```

$x$ is already in that range, but $t \in [0,1]$ is not. Tanh saturates
outside roughly $[-2,2]$ where its derivative vanishes, and inputs on
different scales force the network to learn compensating weight magnitudes.
Putting both on the same scale removes that.

**Boundary points are included in the collocation set.** The 10,000 LHS
points are supplemented with all 456 initial/boundary candidates, giving
10,456 residual points. The PDE holds on the boundary too — nothing about
sitting on an edge exempts a point from $u_t + uu_x - \nu u_{xx} = 0$.
Note the append happens *before* the 100 supervised points are subsampled,
and `np.vstack` copies, so the full 456 are used.

**Supervised points are drawn from a combined pool.** The $t=0$ row and the
$x = \pm 1$ columns give 456 candidates, from which 100 are sampled without
replacement. The initial/boundary split is therefore whatever the draw
gives — roughly 56/44 in expectation — not fixed in advance. This is also
why the loss has a single $\mathrm{MSE}_u$ term: all 100 points are the same
kind of object, a location with a known target.

**No Adam phase.** The paper's forward Burgers case uses L-BFGS alone. (The
inverse/identification version in the same repository does use Adam first,
and the amount depends on the noise level — `model.train(0)` for clean data,
`model.train(10000)` for noisy.)

## Validation of the reference data

`burgers_shock.mat` stores the solution on a $256 \times 100$ grid,
$[x, t]$:

```
data['x'].shape    → (256, 1)
data['t'].shape    → (100, 1)
data['usol'].shape → (256, 100)
```

Checked that the reference solution is actually zero at the Dirichlet
boundaries, since the implementation hardcodes $u_{bc} = 0$ rather than
reading the values from the file as Raissi does:

```
usol[0,:].abs().max()  → 3.57e-16
usol[-1,:].abs().max() → 3.17e-16
```

Zero to machine precision, so the two approaches are equivalent.

## Results

### Default PyTorch initialisation (Kaiming-uniform), 10 seeds

| Seed | Loss | Relative $L^2$ |
|---|---|---|
| 0 | 4.18e-6 | 1.26e-3 |
| 1 | 9.55e-6 | 3.97e-3 |
| 2 | 2.69e-5 | 3.87e-3 |
| 3 | 1.50e-5 | 1.46e-3 |
| 4 | 6.14e-6 | 7.38e-3 |
| 5 | 7.96e-6 | 3.73e-3 |
| 6 | 5.38e-6 | 2.61e-3 |
| 7 | 2.08e-5 | 9.55e-3 |
| 8 | 6.01e-6 | 1.58e-3 |
| 9 | 4.56e-5 | 9.15e-3 |

mean 4.46e-3 · median 3.80e-3 · min 1.26e-3 · max 9.55e-3

### Xavier initialisation, 10 seeds

| Seed | Loss | Relative $L^2$ | Closure calls |
|---|---|---|---|
| 0 | 1.93e-5 | 3.07e-3 | 3861 |
| 1 | 9.48e-6 | 4.74e-3 | 3212 |
| 2 | 4.19e-6 | 7.43e-4 | 5480 |
| 3 | 4.74e-6 | 1.91e-3 | 4082 |
| 4 | 1.76e-6 | **6.86e-4** | 5215 |
| 5 | 7.96e-6 | 1.14e-3 | 5165 |
| 6 | 4.10e-6 | 1.75e-3 | 5002 |
| 7 | 6.80e-6 | 1.39e-3 | 3548 |
| 8 | 3.80e-6 | 4.66e-3 | 5075 |
| 9 | 7.71e-6 | 4.99e-3 | 4274 |

mean 2.51e-3 · median 1.83e-3 · min 6.86e-4 · max 4.99e-3

### Comparison

| | mean | median | min | max |
|---|---|---|---|---|
| Kaiming (PyTorch default) | 4.46e-3 | 3.80e-3 | 1.26e-3 | 9.55e-3 |
| Xavier (matching Raissi) | **2.51e-3** | **1.83e-3** | **6.86e-4** | **4.99e-3** |
| Raissi, reported | — | — | 6.7e-4 | — |


## Running it

Requires `burgers_shock.mat` from
[maziarraissi/PINNs](https://github.com/maziarraissi/PINNs),
under `appendix/Data/`.


## Reference

Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed
neural networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations.* Journal of
Computational Physics 378, 686–707.
[arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

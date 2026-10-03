
# Convection — reproducing Krishnapriyan et al. (2021)

Reproduction of the convection failure mode and the curriculum
regularisation fix from *Characterizing possible failure modes in
physics-informed neural networks* (NeurIPS 2021, arXiv:2109.01050).

## Problem

$$u_t + \beta u_x = 0, \qquad x \in [0, 2\pi],\ t \in [0,1]$$

- **Initial condition:** $u(x,0) = \sin(x)$
- **Boundary:** periodic, $u(0,t) = u(2\pi,t)$
- **Exact solution:** $u(x,t) = \sin(x - \beta t)$


## Why this problem

This is a linear, first order, closed-form solution PDE, one of the easiest. 
However, Krishnapriyan et al. showed that a vanilla PINN nonetheless fails completely for large $\beta$, 
which isolates the cause: not model capacity, not problem complexity, but the optimisation landscape created by the soft
PDE constraint.

Having an analytic solution also means there is no reference-data step, so
any error is attributable to the method rather than to the ground truth.

## Changes from the heat-equation implementation

| | Heat | Convection |
|---|---|---|
| Residual | $u_t - u_{xx}$ | $u_t + \beta u_x$ |
| Domain in $x$ | $[0,1]$ | $[0, 2\pi]$ |
| Initial condition | $\sin(\pi x)$ | $\sin(x)$ |
| Boundary | Dirichlet, $u = 0$ (supervised) | Periodic (unsupervised) |
| Derivatives | $u_t$, $u_{xx}$ | $u_t$, $u_x$ |

The boundary term is the structural difference. A Dirichlet condition has a
known target, so the loss is an ordinary MSE against it. A periodic
condition has no target at all — the network is compared against *itself* at
the two ends:

```python
u_pred_bc_1 = model(torch.cat([torch.zeros(N_b,1), t_bc], dim=1))
u_pred_bc_2 = model(torch.cat([2*torch.pi*torch.ones(N_b,1), t_bc], dim=1))
loss_bc = loss_fn(u_pred_bc_1, u_pred_bc_2)
```

Both ends are evaluated at the same times `t_bc`, and the mismatch is
penalised. This makes the term unsupervised, like the residual.

## Setup

| | |
|---|---|
| Architecture | 5 hidden layers × 50 neurons, tanh |
| Points | $N_{ic} = 100$, $N_b = 100$, $N_f = 5000$, uniform random |
| Loss | $\mathcal{L}_{ic} + \mathcal{L}_{bc} + \mathcal{L}_{pde}$, unweighted |
| Evaluation | 100 × 100 regular grid, relative $L^2$ |

---

## Part 1 — Reproducing the failure mode

Vanilla PINN, trained directly at the target $\beta$, 10,000 Adam epochs.

| $\beta$ | Final loss | Relative $L^2$ |
|---|---|---|
| 1 | 4.1e-4 | 6.7e-3 |
| 30 | 3.8e-2 | **8.7e-1** |

Identical architecture, loss, optimiser and iteration count in both cases.
An error of 0.87 is close to the 1.0 obtained by predicting $u \equiv 0$
everywhere, so this is near-total failure rather than degraded accuracy —
and the $\beta = 30$ solution is no more complex than the $\beta = 1$ one.

Failure mode reproduced.

---

## Part 2 — Curriculum regularisation

Instead of training directly at the target, train through an increasing
sequence of $\beta$, warm-starting each stage from the previous one's
weights.

There is no explicit save/load step: the model is constructed **once**
outside the loop, and `optimizer.step()` mutates it in place, so stage
$n+1$ simply begins where stage $n$ ended.

```python
model = NeuralNetwork()
for beta in [...]:
    train_model(model, beta, epochs=...)
```

Evidence the warm start is working: the epoch-0 loss at each new stage is
large and generally increasing (0.48 → 7.94 → 12.42 → 51.16), because the
previous stage's weights solve a *different* equation. A freshly
initialised network would not produce that pattern.

### Run 1 — coarse schedule, 2000 epochs/stage

$\beta \in \{1, 5, 10, 20, 30\}$

| $\beta$ | Final loss | Relative $L^2$ |
|---|---|---|
| 1 | 9.3e-5 | 9.1e-3 |
| 5 | 3.6e-4 | 1.6e-2 |
| 10 | 8.5e-4 | 4.6e-2 |
| 20 | 1.0e-1 | **8.9e-1** |
| 30 | 6.4e-2 | 9.6e-1 |

Holds through $\beta = 10$, collapses at $\beta = 20$. The final result is
no better than the from-scratch baseline.

### Run 2 — finer schedule, 2000 epochs/stage

$\beta \in \{1, 5, 10, 15, 20, 25, 30\}$

| $\beta$ | Final loss | Relative $L^2$ |
|---|---|---|
| 1 | 2.5e-4 | 3.6e-2 |
| 5 | 3.4e-3 | 3.3e-2 |
| 10 | 5.7e-4 | 3.6e-2 |
| 15 | 3.3e-2 | **4.3e-1** |
| 20 | 3.7e-3 | 1.1e-1 |
| 30 | 2.4e-2 | 7.6e-1 |

Finer steps did not fix it. 
### Run 3 — finer schedule, 10000 epochs/stage

| $\beta$ | Final loss | Relative $L^2$ |
|---|---|---|
| 1 | 1.2e-4 | 2.4e-2 |
| 5 | 2.9e-5 | 5.7e-3 |
| 10 | 1.1e-4 | 1.3e-2 |
| 15 | 5.5e-5 | 1.6e-2 |
| 20 | 1.3e-3 | 4.6e-2 |
| 25 | 1.4e-4 | 3.3e-2 |
| 30 | 1.2e-3 | **3.2e-2** |

The error now stays bounded across the whole schedule — no collapse. 

### Run 4 — Adam + L-BFGS per stage

Two-phase optimisation, following the recipe used in both Raissi et al. and
Krishnapriyan et al.: Adam first to reach a good region, then L-BFGS to
converge deeply.

Adam is first-order — gradient only, with per-parameter adaptive step sizes
and momentum. Robust from random initialisation, but it plateaus because it
carries no curvature information. L-BFGS is quasi-Newton: it estimates
second-derivative information from how successive gradients change, giving
far better step directions and lengths. It converges much deeper but needs
full-batch gradients and a reasonable starting point, so it runs after Adam
rather than instead of it.

Implementation notes:

- The loss is factored into a `compute_loss()` function, called by both the
  Adam loop and the L-BFGS closure.
- L-BFGS needs a `closure` because its line search evaluates the loss
  several times per step. `lbfgs.step(closure)` is called **once**;
  `max_iter` controls the internal iterations.
- Optimisers are constructed inside `train_model`, so each stage gets fresh
  optimiser state. Previously Adam's momentum and second-moment estimates
  carried across a coefficient change, having been accumulated for a
  different loss landscape.

| | |
|---|---|
| Adam | 5,000 epochs, lr $10^{-3}$ |
| L-BFGS | `max_iter` 2,000, lr 1.0, `strong_wolfe` line search |
| Schedule | $\beta \in \{1, 5, 10, 15, 20, 25, 30\}$ |

| $\beta$ | After Adam | After L-BFGS | Relative $L^2$ |
|---|---|---|---|
| 1 | 2.7e-4 | 1.9e-5 | 5.1e-3 |
| 5 | 6.6e-4 | 2.1e-6 | 8.8e-4 |
| 10 | 9.6e-5 | 7.3e-6 | 2.1e-3 |
| 15 | 2.7e-3 | 1.9e-5 | 8.0e-3 |
| 20 | 4.5e-2 | 1.2e-5 | 5.2e-3 |
| 25 | 1.9e-2 | 1.9e-5 | 1.0e-2 |
| 30 | 1.5e-3 | 4.7e-5 | **1.7e-2** |

---

## Summary at $\beta = 30$

| Method | Epochs/stage | Relative $L^2$ |
|---|---|---|
| Vanilla PINN, trained directly | 10,000 total | 8.7e-1 |
| Curriculum, coarse schedule, Adam | 2,000 | 9.6e-1 |
| Curriculum, fine schedule, Adam | 2,000 | 7.6e-1 |
| Curriculum, fine schedule, Adam | 10,000 | 3.2e-2 |
| Curriculum, fine schedule, Adam + L-BFGS | 5,000 + 2,000 | **1.7e-2** |

51× improvement over the vanilla baseline, at half the epoch budget of
run 3.

## Caveats

Results are single runs, not averaged over seeds. The run-to-run spread at
fixed settings is large — $\beta = 1$ gave $6.7\text{e-}3$, $9.1\text{e-}3$
and $3.6\text{e-}2$ across different runs of identical code — so individual
numbers here should be read as indicative rather than precise. The Burgers
section of this repo reports a 10-seed distribution instead.

## Reference

Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R. & Mahoney, M. W. (2021).
*Characterizing possible failure modes in physics-informed neural networks.*
NeurIPS. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)

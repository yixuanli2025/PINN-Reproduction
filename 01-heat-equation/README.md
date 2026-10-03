# PINN for the 1D Heat Equation

### Problem

$$u_t = u_{xx}, \qquad x \in [0,1],\ t \in [0,1]$$

**Initial condition:** $u(x,0) = \sin(\pi x)$

**Boundary conditions:** $u(0,t) = u(1,t) = 0$

**Exact solution:** $u(x,t) = \sin(\pi x)\, e^{-\pi^2 t}$

### Why this problem:

A validation step before reproducing Raissi et al. (2019). Since the exact solution is known analytically, any error stem directly from code implementation rather than to the reference data or the difficulty of the equation.

### Method

The network is the candidate solution: $u_\theta(x,t)$, mapping $(x,t) \mapsto u$. The loss has three terms:

$$\mathcal{L} = \underbrace{\frac{1}{N_0}\sum \big(\hat u - u_0\big)^2}_{\text{initial condition}} + \underbrace{\frac{1}{N_b}\sum \hat u^2}_{\text{boundary}} + \underbrace{\frac{1}{N_f}\sum \big(\hat u_t - \hat u_{xx}\big)^2}_{\text{PDE residual}}$$

Derivatives $\hat u_t$, $\hat u_{xx}$ are obtained by automatic differentiation of the network output with respect to its inputs.

### Setup

|              |                                                         |
| ------------ | ------------------------------------------------------- |
| Architecture | 5 hidden layers × 50 neurons, tanh                      |
| Points       | $N_0 = 100$, $N_b = 100$, $N_f = 5000$ (uniform random) |
| Optimizer    | Adam, lr $= 10^{-3}$                                    |
| Evaluation   | 100 × 100 regular grid, relative $L^2$ error            |

note that for the heat equation PINN: 

	- only Adam optimizer is used
	- no input normalization
	- PyTorch's default initialization

### Results

| Run | Iterations | Final loss         | Relative $L^2$ error |
| --- | ---------- | ------------------ | -------------------- |
| 1   | 5,000      | $2.3\times10^{-4}$ | $5.8\times10^{-2}$   |
| 2   | 10000      | $1.8\times10^{-5}$ | $7.4\times10^{-3}$   |

### Diagnosis

Run 1's loss was small ($2.3\times10^{-4}$) while the error was large
($5.8\times10^{-2}$). I suspected two candidates: undertraining, or imbalance contribution of the three loss terms, where one dominates and the others are effectively ignored.

Printing the terms separately at 10,000 iterations:

| Term | Value |
|---|---|
| $\mathcal{L}_{ic}$ | 2.6e-6 |
| $\mathcal{L}_{bc}$ | 2.1e-6 |
| $\mathcal{L}_{pde}$ | 7.9e-6 |

All within a factor of four, so imbalance is ruled out. Doubling the
iterations cut the error ~8×, confirming undertraining.

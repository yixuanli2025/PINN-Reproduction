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

### Best Results

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

### Note on reproducibility

The two runs above come from different random draws — no seed was fixed at
this stage. Re-running the same code gives results of the same order but
not the same numbers.

| Run | Iterations | Final loss | Relative $L^2$ |
|---|---|---|---|
| 1 | 5,000 | 2.3e-4 | 5.8e-2 |
| 2 | 10,000 | 1.8e-5 | 7.4e-3 |
| 3 | 10,000 | 1.1e-3 | — |
| 4 | 10,000 | 2.5e-5 | 1.2e-2 |

#### The full loss traces for #3 #4

**Run 3**

| Epoch | Loss     | Epoch | Loss     |
| ----- | -------- | ----- | -------- |
| 0     | 4.86e-01 | 5000  | 9.74e-05 |
| 500   | 5.77e-03 | 5500  | 3.54e-04 |
| 1000  | 2.21e-03 | 6000  | 8.14e-04 |
| 1500  | 5.65e-04 | 6500  | 4.64e-04 |
| 2000  | 3.82e-04 | 7000  | 3.65e-04 |
| 2500  | 1.07e-03 | 7500  | 6.11e-05 |
| 3000  | 5.20e-04 | 8000  | 1.21e-03 |
| 3500  | 3.69e-04 | 8500  | 5.01e-05 |
| 4000  | 7.77e-04 | 9000  | 1.99e-04 |
| 4500  | 1.61e-04 | 9500  | 1.05e-04 |

Final terms: 
$\mathcal{L}_{ic} = 1.36\times10^{-4}$,
$\mathcal{L}_{bc} = 1.57\times10^{-5}$,
$\mathcal{L}_{pde} = 8.43\times10^{-4}$

**Run 4**

| Epoch | Loss     | Epoch | Loss     |
| ----- | -------- | ----- | -------- |
| 0     | 4.66e-01 | 5000  | 6.53e-05 |
| 500   | 1.12e-02 | 5500  | 6.19e-05 |
| 1000  | 8.73e-04 | 6000  | 4.76e-05 |
| 1500  | 4.53e-04 | 6500  | 2.77e-04 |
| 2000  | 3.13e-04 | 7000  | 4.20e-05 |
| 2500  | 2.61e-04 | 7500  | 5.65e-05 |
| 3000  | 1.70e-04 | 8000  | 4.13e-04 |
| 3500  | 1.34e-04 | 8500  | 2.21e-05 |
| 4000  | 1.03e-04 | 9000  | 8.21e-05 |
| 4500  | 8.29e-05 | 9500  | 1.93e-05 |

Final terms: $\mathcal{L}_{ic} = 3.10\times10^{-6}$,
$\mathcal{L}_{bc} = 7.11\times10^{-6}$,
$\mathcal{L}_{pde} = 1.52\times10^{-5}$

Note the jumps: run 3 goes $6.11\times10^{-5} \to 1.21\times10^{-3}$
between epochs 7500 and 8000, a factor of 20; run 4 goes
$4.76\times10^{-5} \to 2.77\times10^{-4}$ at 6500.

The difficulty in converging likely arises from the optimization dynamics of Adam, producing the overshoot-and-recover pattern visible in the epoch–loss tables. In particular, the fixed learning rate of $\mathrm{lr}=10^{-3}$ may be too large as training approaches a local minimum.

The later sections (convection, Burgers) address this: a seed is fixed,
L-BFGS replaces or follows Adam, and results are reported as a distribution
over 10 runs rather than a single number.

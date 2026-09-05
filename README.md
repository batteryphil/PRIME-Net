# PRIME-Net: Pareto-Refined Invariant Mining Engine

**PRIME-Net** is an automated **Symbolic Regression & Invariant Mining Engine** built on discrete program synthesis. Rather than training continuous black-box neural networks that memorize training distributions through overparameterized curve-fitting, PRIME-Net discovers human-interpretable mathematical equations directly from empirical telemetry.

---

## Key Architectural Pillars

1. **Age-Fitness Pareto Optimization (AFPO)**:
   Abandons unimodal gradient-based optimization in favor of multi-objective Pareto optimization across Age and Mean Squared Error. By tracking the genotypic age of candidate mathematical structures, newly composed expressions are protected from premature extinction by ancient local minima, effectively eliminating "tissue rejection" in modular crossover.

2. **Numba JIT Stack Evaluator**:
   Equations are represented as Reverse Polish Notation (RPN) integer arrays evaluated via an ultra-fast, protected LLVM CPU kernel (`@njit`). It evaluates populations of thousands of candidate equations in milliseconds with inline numerical guards against singularities and domain errors.

3. **Hierarchical Subtree Library ("Parts Bin")**:
   When candidate expressions exhibit high optimization potential, valid subtrees are extracted, canonically hashed using Abstract Syntax Trees (ASTs), and mapped into dynamic macro-tokens. This enables the engine to cross combinatorial search barriers and assemble nested physical laws without exponential search degradation.

4. **Non-Linear Minimum Description Length (MDL) Penalty**:
   A structural complexity penalty ($L \log_2(L + 1)$) acts as an algorithmic implementation of **Occam's Razor**, penalizing unrolled token length and operator stacking unless accompanied by an exponential decrease in empirical loss. This strictly rejects sensor noise and prevents overfitting.

---

## Installation

```bash
git clone https://github.com/batteryphil/PRIME-Net.git
cd PRIME-Net
pip install -r requirements.txt
```

---

## Quick Start

### 1. Functional Interface (`run_prime_engine`)

```python
import numpy as np
from prime_core import run_prime_engine

# Prepare telemetry
X_train = np.random.uniform(-1, 1, (1000, 3))
y_train = np.sin(X_train[:, 0]) * np.exp(-X_train[:, 1])

# Run symbolic discovery
res = run_prime_engine(X_train, y_train, pop_size=512, max_generations=1000)

print(f"Discovered Equation : {res['best_sympy']}")
print(f"Train R^2 Score     : {res['train_r2']:.4f}")
print(f"Train MSE           : {res['train_mse']:.6f}")
```

### 2. Scikit-Learn Compatible Interface (`PrimeRegressor`)

```python
import numpy as np
from prime_core import PrimeRegressor

X = np.random.uniform(-2, 2, (500, 2))
y = 2.0 * X[:, 0] + X[:, 1]**2

# Initialize estimator
model = PrimeRegressor(pop_size=256, max_generations=1500)
model.fit(X, y)

print(f"Learned Formula : {model.equation_}")
print(f"Model R^2 Score : {model.score(X, y):.4f}")

# Predict on new data
y_pred = model.predict(X[:5])
```

---

## Empirical Benchmarks & Validation

PRIME-Net has been evaluated across disparate scientific, mathematical, and noisy engineering domains:

### 1. Mathematical Scaling Laws & Number Theory
* **Problem:** Asymptotic behavior of the prime counting function $\pi(x)$.
* **Discovery:** Recovered Legendre's approximation and the logarithmic density scaling $\frac{x}{\ln(x)}$.
* **Takeaway:** Demonstrated convergence across discrete, non-differentiable loss landscapes where gradient-based methods fail.

### 2. Neural System Identification (Mamba-130M Latent Geometry)
* **Problem:** Extracting analytical dynamical laws governing the 25,600-dimensional recurrent hidden states of a Mamba State Space Model.
* **Discovery:** Compact non-linear surrogates (e.g., $- \sin(X_1 - X_{10})$) mapped directly on PCA-bottlenecked latent manifolds.
* **Takeaway:** Bridges continuous latent embeddings to symbolic, human-readable governing differential rules.

### 3. Fluid Dynamics & Convective Acceleration (Navier-Stokes)
* **Problem:** Predicting convective acceleration $u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}$ from velocity vector fields in simulated Taylor-Green vortex flows.
* **Discovery:** Accurately reconstructed product-sum combinations matching the momentum equations.

### 4. Epidemiology & Non-Linear Biological Transmission
* **Problem:** Reconstructing non-linear transmission dynamics ($\beta I S$) from simulated hospital case telemetry corrupted with 20% Gaussian observation noise.
* **Discovery:** Isolated the core underlying infection rate while filtering high-frequency reporting noise.

### 5. False-Positive Suppression & Noise Rejection (Low SNR Regimes)
* **Problem:** Evaluating datasets with high-amplitude noise floor and negligible underlying structural signal (e.g., faint signals buried in Gaussian noise, tectonic frequency correlations).
* **Discovery:** The engine consistently converges to the null model (constant 0) rather than inventing hallucinated terms.
* **Takeaway:** Strict MDL complexity penalization provides robust resistance to pareidolia and spurious correlation discovery.

---

## Running the Automated Test Suite

```bash
python test_quickstart.py
```

---

*Developed as part of the Antigravity Project.*


import time
import numpy as np
from test_integrated_improvements import run_experiment

benchmarks = [
    {
        "name": "Poly (X0^2 + X1^2)",
        "fn": lambda X: X[:, 0]**2 + X[:, 1]**2,
        "n_vars": 2
    },
    {
        "name": "Continuous Scaled (3.5*sin(X0) - 1.25*X1)",
        "fn": lambda X: 3.5 * np.sin(X[:, 0]) - 1.25 * X[:, 1],
        "n_vars": 2
    },
    {
        "name": "Rational (X0 / (1 + X1^2))",
        "fn": lambda X: X[:, 0] / (1.0 + X[:, 1]**2),
        "n_vars": 2
    },
    {
        "name": "Exp (exp(-X0^2) + X1)",
        "fn": lambda X: np.exp(-X[:, 0]**2) + X[:, 1],
        "n_vars": 2
    },
    {
        "name": "4D Product (X0*X1 + X2*X3)",
        "fn": lambda X: X[:, 0]*X[:, 1] + X[:, 2]*X[:, 3],
        "n_vars": 4
    }
]

print(f"{'Benchmark':<35} | {'Base R2':<10} | {'Impr R2':<10} | {'Base MSE':<10} | {'Impr MSE':<10} | {'Winner'}")
print("-" * 95)

results = []
np.random.seed(999)

for bm in benchmarks:
    X = np.random.uniform(-1.5, 1.5, (400, bm["n_vars"]))
    y = bm["fn"](X)

    res_b = run_experiment(X, y, mode="baseline", max_generations=40, pop_size=128, seed=42)
    res_i = run_experiment(X, y, mode="improved", max_generations=40, pop_size=128, seed=42)

    winner = "IMPROVED" if res_i["r2"] > res_b["r2"] + 1e-4 else ("BASELINE" if res_b["r2"] > res_i["r2"] + 1e-4 else "TIE")
    print(f"{bm['name']:<35} | {res_b['r2']:<10.4f} | {res_i['r2']:<10.4f} | {res_b['best_mse']:<10.4f} | {res_i['best_mse']:<10.4f} | {winner}")

    results.append({
        "name": bm["name"],
        "base_r2": res_b["r2"],
        "impr_r2": res_i["r2"],
        "base_mse": res_b["best_mse"],
        "impr_mse": res_i["best_mse"],
    })

import pandas as pd
import numpy as np
import sympy
from scipy.optimize import minimize
from prime_core import PrimeRegressor
from tune_prime import test_config

print("=== PRIME-Net Self-Improving Meta-Optimizer ===")

# 1. Gather initial random exploration data
print("\nPhase 1: Initial Exploration (Random Search)")
history = []
for _ in range(5):
    mut = np.random.randint(1000, 4000)
    step = np.random.randint(500, 4000)
    flip = np.random.uniform(0.05, 0.2)
    print(f"Testing Mut={mut}, Step={step}, Flip={flip:.3f}...")
    fit, gen = test_config(mut, step, flip, max_gens=15)
    history.append([mut, step, flip, fit])

best_baseline_fit = max([h[3] for h in history])
print(f"Best initial baseline fitness: {best_baseline_fit:.1f}")

# 2. Meta-Learning Loop
for meta_epoch in range(3):
    print(f"\nPhase 2: Meta-Learning Epoch {meta_epoch+1}")
    
    # Prepare Dataset
    data = np.array(history)
    X = data[:, :3]
    y = data[:, 3]
    
    # Normalize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    print("Running Symbolic Regression on Performance Data...")
    model = PrimeRegressor(pop_size=500, max_generations=1000, timeout_sec=15.0)
    model.fit(X_norm, y)
    
    best_eq_str = model.equation_
    print(f"Discovered Law: Fitness = {best_eq_str}")
    
    if best_eq_str is None or "nan" in str(best_eq_str).lower():
        print("Invalid equation found, exploring randomly...")
        mut = np.random.randint(1000, 4000)
        step = np.random.randint(500, 4000)
        flip = np.random.uniform(0.05, 0.2)
    else:
        # 3. Mathematical Optimization (Calculus)
        # We want to MAXIMIZE the discovered formula.
        x1, x2, x3 = sympy.symbols('X1 X2 X3')
        
        # Safe evaluation
        try:
            expr = sympy.sympify(best_eq_str)
            f_lamb = sympy.lambdify((x1, x2, x3), expr, modules=['numpy'])
            
            # Objective: minimize negative fitness
            def objective(x):
                try:
                    val = f_lamb(x[0], x[1], x[2])
                    if np.isnan(val) or np.isinf(val): return 1e9
                    return -float(val)
                except:
                    return 1e9
            
            # Start search from the best known point
            best_idx = np.argmax(y)
            x0 = X_norm[best_idx]
            
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100})
            
            # Denormalize predicted optimal parameters
            best_x_norm = res.x
            best_x = best_x_norm * X_std + X_mean
            
            mut = int(np.clip(best_x[0], 100, 10000))
            step = int(np.clip(best_x[1], 100, 10000))
            flip = float(np.clip(best_x[2], 0.01, 0.5))
            print(f"Math Optimizer suggests new parameters: Mut={mut}, Step={step}, Flip={flip:.3f}")
        except Exception as e:
            print(f"Math Optimizer failed ({e}), exploring randomly...")
            mut = np.random.randint(1000, 4000)
            step = np.random.randint(500, 4000)
            flip = np.random.uniform(0.05, 0.2)
    
    # 4. Evaluate the new mathematically-discovered configuration
    print(f"Evaluating mathematically proposed configuration...")
    fit, gen = test_config(mut, step, flip, max_gens=15)
    print(f"Resulting Fitness: {fit:.1f}")
    history.append([mut, step, flip, fit])

print("\n=== Meta-Optimization Complete ===")
final_best_fit = max([h[3] for h in history])
print(f"Baseline Fitness: {best_baseline_fit:.1f}")
print(f"Meta-Learned Fitness: {final_best_fit:.1f}")
improvement = final_best_fit - best_baseline_fit
print(f"Improvement over baseline: +{improvement:.1f}")

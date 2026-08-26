"""
PRIME Architecture - Phase C: Critical Boundary Mapping
This script performs a hyperparameter sweep over the Occam's Razor penalty (lambda)
to identify the exact boundaries between Structural Hallucination (Overfitting), 
Perfect Recovery (The Goldilocks Zone), and Local Minima Traps (Underfitting).

Test Subject: Newton's Law of Universal Gravitation (F = m1*m2 / r^2)
Target Structure: (X1 * X2) / (X3 * X3)
"""
import numpy as np
import time
import csv
from prime_benchmark_matrix import run_prime_numba

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME ARCHITECTURE: CRITICAL BOUNDARY MAPPING (LAMBDA SWEEP)")
    print("================================================================================\n")
    
    np.random.seed(42)
    N = 1000
    X_val = np.random.uniform(1, 10, (N, 3)).astype(np.float32)
    m1 = X_val[:, 0]
    m2 = X_val[:, 1]
    r  = X_val[:, 2]
    
    # Y = (m1 * m2) / r^2
    Y_val = ((m1 * m2) / (r * r)).astype(np.float32)
    
    X_full = np.zeros((N, 6), dtype=np.float32)
    X_full[:, :3] = X_val
    
    # Lambda Values to Sweep
    # Expected:
    # 0.0 -> Overfit (massive hallucinated structure)
    # 0.005+ -> Underfit (local minimum like log(X1)/X3)
    # Somewhere in between -> Perfect 7-token structural recovery!
    lambda_sweep = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
    generations = 5000
    
    print(f"Sweeping {len(lambda_sweep)} lambda values for {generations} generations each...")
    print("Target Structure: (X1 * X2) / (X3 * X3)")
    print("--------------------------------------------------------------------------------")
    
    with open('lambda_sweep_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda', 'Final_Equation', 'Complexity', 'Test_MSE', 'Time_s'])
        
        for lmb in lambda_sweep:
            print(f"[*] Testing Lambda = {lmb:.4f} ... ", flush=True)
            
            best_eq = ""
            best_comp = 0
            best_te_mse = float('inf')
            total_dur = 0
            
            # Multi-start optimization (3 seeds) to escape local minima
            for seed in [42, 1337, 9999]:
                print(f"    -> Seed {seed} ... ", end="", flush=True)
                eq, comp, tr_mse, te_mse, dur, telemetry = run_prime_numba(
                    name=f"Sweep_{lmb}_{seed}", 
                    X_full=X_full, 
                    Y_full=Y_val, 
                    lambda_penalty=lmb, 
                    generations=generations,
                    seed=seed
                )
                print(f"{dur:.1f}s | MSE: {te_mse:.5f}")
                total_dur += dur
                if te_mse < best_te_mse:
                    best_te_mse = te_mse
                    best_eq = eq
                    best_comp = comp
            
            print(f"  [Best for {lmb:.4f}] Eq: {best_eq}")
            print(f"  Complexity: {best_comp} | Test MSE: {best_te_mse:.8f}\n")
            
            writer.writerow([lmb, best_eq, best_comp, best_te_mse, total_dur])
            f.flush()
            
    print("Sweep complete! Data saved to lambda_sweep_results.csv")

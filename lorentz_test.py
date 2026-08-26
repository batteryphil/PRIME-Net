"""
PRIME Architecture - Famous Physics Test: Special Relativity
Target: Discover the Lorentz Factor (gamma) from Special Relativity
Formula: gamma = 1 / sqrt(1 - v^2)

Since our vocabulary DOES NOT possess a 'sqrt' or 'pow' operator, 
PRIME must independently discover the mathematical identity:
1 / sqrt(x) = exp(-0.5 * log(x))
and apply it to (1 - v^2).
"""
import numpy as np
from prime_benchmark_matrix import run_prime_numba

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME ARCHITECTURE: SPECIAL RELATIVITY (LORENTZ FACTOR)")
    print("================================================================================\n")
    
    # X1 = velocity (v), normalized to fractions of c (speed of light)
    # v ranges from 0.01 to 0.99
    np.random.seed(42)
    N = 1000
    X_val = np.random.uniform(0.01, 0.99, N).astype(np.float32)
    
    # Y = Lorentz Factor: 1 / sqrt(1 - v^2)
    Y_val = 1.0 / np.sqrt(1.0 - X_val**2).astype(np.float32)
    
    # Format for PRIME (N, 6) where only column 0 is used
    X_full = np.zeros((N, 6), dtype=np.float32)
    X_full[:, 0] = X_val
    
    print(f"Dataset: 1000 relativistic velocities [0.01c, 0.99c]")
    print("Target: The Lorentz Factor y = 1 / sqrt(1 - v^2)")
    print("Constraint: The engine has no 'sqrt' operator in its vocabulary.")
    print("Goal: Discover the fractional power identity via exp and log.\n")
    
    print("Running PRIME Numba JIT Optimizer (Generations: 10000)...")
    eq, comp, tr_mse, te_mse, dur = run_prime_numba(
        name="Lorentz_Factor", 
        X_full=X_full, 
        Y_full=Y_val, 
        lambda_penalty=0.005, 
        generations=10000
    )
    
    print("\n--------------------------------------------------------------------------------")
    print(f"  Final Recovered Equation: {eq}")
    print(f"  Complexity: {comp}")
    print(f"  Test MSE:   {te_mse:.5f}")
    print(f"  Compute Time: {dur:.1f}s")
    print("--------------------------------------------------------------------------------")

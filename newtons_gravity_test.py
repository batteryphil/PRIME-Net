"""
PRIME Architecture - Famous Physics Test: Newton's Law of Universal Gravitation
Target: Discover the Inverse Square Law from raw mass and distance data.
Formula: F = (m1 * m2) / r^2

This is a classic test for symbolic regression. The engine must perfectly isolate
the multiplicative relationship between the masses in the numerator, and the
exponential decay (inverse square) of the distance in the denominator.
"""
import numpy as np
import time
from prime_benchmark_matrix import run_prime_numba

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME ARCHITECTURE: NEWTON'S LAW OF UNIVERSAL GRAVITATION")
    print("================================================================================\n")
    
    # Generate 1000 random physics interactions
    np.random.seed(42)
    N = 1000
    
    # X1 = Mass 1 (m1), ranging from 1 to 10
    # X2 = Mass 2 (m2), ranging from 1 to 10
    # X3 = Distance (r), ranging from 1 to 10
    X_val = np.random.uniform(1, 10, (N, 3)).astype(np.float32)
    
    m1 = X_val[:, 0]
    m2 = X_val[:, 1]
    r  = X_val[:, 2]
    
    # Y = Gravitational Force: (m1 * m2) / r^2
    # Note: We normalize the gravitational constant G to 1.0 for structural recovery
    Y_val = ((m1 * m2) / (r * r)).astype(np.float32)
    
    # Format for PRIME (N, 6) where cols 0,1,2 are used
    X_full = np.zeros((N, 6), dtype=np.float32)
    X_full[:, :3] = X_val
    
    print(f"Dataset: 1000 celestial body interactions.")
    print("Variables: X1 (Mass 1), X2 (Mass 2), X3 (Distance r)")
    print("Target: The Inverse Square Law F = (X1 * X2) / (X3 * X3)")
    print("Goal: Perfectly reconstruct the exact algebraic structure of Gravity.\n")
    
    # Give it 20,000 generations to find the perfect structure
    print("Running PRIME Numba JIT Optimizer (Generations: 20000)...")
    eq, comp, tr_mse, te_mse, dur = run_prime_numba(
        name="Newton_Gravity", 
        X_full=X_full, 
        Y_full=Y_val, 
        lambda_penalty=0.001,  # Lower penalty to allow slightly deeper equations (like r*r)
        generations=20000
    )
    
    print("\n--------------------------------------------------------------------------------")
    print(f"  Final Recovered Equation: {eq}")
    print(f"  Complexity: {comp}")
    print(f"  Test MSE:   {te_mse:.8f}")
    print(f"  Compute Time: {dur:.1f}s")
    print("--------------------------------------------------------------------------------")

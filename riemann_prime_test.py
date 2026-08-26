"""
PRIME Architecture - Prime Number Theorem / Riemann Zeta Test
This script generates a dataset based on the Prime Counting Function pi(x),
which is deeply connected to the Riemann Hypothesis.
The Prime Number Theorem states that pi(x) ~ x / ln(x).
We will see if PRIME can discover this fundamental mathematical truth
from raw data!
"""
import numpy as np
import time
from prime_benchmark_matrix import run_prime_numba, rpn_to_str

def generate_primes(n):
    """Sieve of Eratosthenes to generate primes up to n."""
    sieve = np.ones(n // 2, dtype=bool)
    for i in range(3, int(n**0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i*i // 2::i] = False
    prime_indices = 2 * np.nonzero(sieve)[0][1:].astype(np.int32) + 3
    return np.concatenate(([2], prime_indices))

def prime_counting_function(x_vals, primes):
    """Returns pi(x) for each x in x_vals."""
    y_vals = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        y_vals[i] = np.searchsorted(primes, x, side='right')
    return y_vals

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME ARCHITECTURE: PRIME NUMBER THEOREM (RIEMANN CONNECTION) TEST")
    print("================================================================================\n")
    
    print("Generating Prime Counting Function (pi(x)) dataset...")
    MAX_VAL = 10000
    primes = generate_primes(MAX_VAL * 2)
    
    # Generate 1000 random points between 10 and MAX_VAL
    np.random.seed(42)
    X_val = np.random.uniform(10, MAX_VAL, 1000).astype(np.float32)
    Y_val = prime_counting_function(X_val, primes).astype(np.float32)
    
    # Format for PRIME (N, 6) where only column 0 is used
    X_full = np.zeros((1000, 6), dtype=np.float32)
    X_full[:, 0] = X_val
    
    print(f"Dataset ready. X range: [10, {MAX_VAL}]. Target: pi(x)")
    print("Goal: PRIME should discover the Prime Number Theorem: pi(x) ~ x / log(x)\n")
    
    # We increase generations to 50,000 to see if it finds an even tighter bound
    print("Running PRIME Numba JIT Optimizer (Generations: 50000)...")
    eq, comp, tr_mse, te_mse, dur = run_prime_numba(
        name="Prime_Theorem", 
        X_full=X_full, 
        Y_full=Y_val, 
        lambda_penalty=0.01, 
        generations=50000
    )
    
    print("\n--------------------------------------------------------------------------------")
    print(f"  Final Recovered Equation: {eq}")
    print(f"  Complexity: {comp}")
    print(f"  Test MSE:   {te_mse:.2f}")
    print(f"  Compute Time: {dur:.1f}s")
    print("--------------------------------------------------------------------------------")
    
    # Compare with true PNT: x / log(x)
    pnt_pred = X_val / np.log(X_val)
    pnt_mse = np.mean((pnt_pred - Y_val)**2)
    print(f"\nFor reference, the theoretical Prime Number Theorem (x / log(x))")
    print(f"achieves an MSE of: {pnt_mse:.2f} on this dataset.")

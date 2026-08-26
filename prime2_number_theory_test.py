import time
import numpy as np
import sympy as sp
from prime_core import run_prime_engine

def generate_primes(n_primes):
    print(f"[*] Generating the first {n_primes} primes using sympy...")
    # sympy.primerange(a, b) generates primes in [a, b)
    # the 100,000th prime is 1,299,709. 
    # To be safe, generate primes up to 1,500,000 and slice
    primes = list(sp.primerange(2, 1500000))
    return np.array(primes[:n_primes], dtype=np.float64)

def run_prime_experiment():
    N_PRIMES = 100000
    primes = generate_primes(N_PRIMES)
    
    n_idx = np.arange(1, N_PRIMES + 1, dtype=np.float64)
    ln_n = np.log(n_idx)
    n_ln_n = n_idx * ln_n
    
    # Avoid log(log(1)) which is log(0) = -inf
    ln_ln_n = np.log(np.where(ln_n > 0, ln_n, 1e-9))
    
    # Input matrix X: [n, ln(n), ln(ln(n)), n/ln(n)]
    X_matrix = np.column_stack([n_idx, ln_n, ln_ln_n, n_idx / np.where(ln_n > 0, ln_n, 1.0)])
    
    # Target: Predict P(n) directly
    Y_vector = primes
    
    # Train-Test Split (First 50k train, next 50k test)
    split_idx = 50000
    
    X_train = X_matrix[:split_idx]
    y_train = Y_vector[:split_idx]
    
    X_test = X_matrix[split_idx:]
    y_test = Y_vector[split_idx:]
    
    print(f"[*] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[*] X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Normalize features to prevent overflow during symbolic evaluation
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 127,
        'macro_seq_len': 31,
        'max_generations': 5000,
        'timeout_sec': 120.0 # Time limit to prevent infinite run in sandbox
    }
    
    print("[*] Launching PRIME 2.0 Engine...")
    t0 = time.time()
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    elapsed = time.time() - t0
    
    print(f"\n=== PRIME 2.0 EVALUATION RESULTS ===")
    print(f"Time Elapsed    : {elapsed:.2f}s")
    print(f"Discovered Eq   : {res['best_sympy']}")
    print(f"Train MSE       : {res['train_r2']:.6f} (Normalized)")
    
    # The current engine returns train_r2 as the test_r2 (simplified).
    # Let's manually calculate the true Test MSE using the returned equation.
    print("[*] Evaluating True Test Generalization...")
    if res['best_sympy'] is not None:
        expr = sp.sympify(res['best_sympy'])
        symbols = [sp.Symbol(f'X{i+1}') for i in range(X_train.shape[1])]
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        inputs_test = [X_test_norm[:, i] for i in range(X_test_norm.shape[1])]
        try:
            y_pred_test_norm = f_eval(*inputs_test)
            valid_mask = ~np.isnan(y_pred_test_norm) & ~np.isinf(y_pred_test_norm)
            
            if np.sum(valid_mask) > 0:
                test_mse_norm = np.mean((y_test_norm[valid_mask] - y_pred_test_norm[valid_mask])**2)
                
                # Denormalize to get real prime gap error
                y_pred_test = y_pred_test_norm * y_std + y_mean
                y_true_test = y_test_norm * y_std + y_mean
                test_mse_true = np.mean((y_true_test[valid_mask] - y_pred_test[valid_mask])**2)
                test_mae_true = np.mean(np.abs(y_true_test[valid_mask] - y_pred_test[valid_mask]))
                
                print(f"Test MSE (Norm) : {test_mse_norm:.6f}")
                print(f"Test MSE (True) : {test_mse_true:.2f}")
                print(f"Test MAE (True) : {test_mae_true:.2f} (Average error in estimating the prime)")
            else:
                print("Test MSE: inf (Equation produced all NaNs on test set)")
                
        except Exception as e:
            print(f"Error evaluating test set: {e}")
            
if __name__ == "__main__":
    run_prime_experiment()

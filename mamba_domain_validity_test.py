import sympy as sp
import numpy as np
import pandas as pd
import warnings
from mamba_data_loader import extract_mamba_data

# Suppress runtime warnings for expected math domain errors (log of negative number, etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_complexity(expr_str):
    # Rough approximation of symbolic complexity (count of operators/operands)
    expr = sp.sympify(expr_str)
    return sp.count_ops(expr) + len(expr.free_symbols)

def run_domain_validity_test():
    print("[*] Generating reference latent trajectory...")
    X_matrix, Y_matrix = extract_mamba_data(T=256)
    
    X_mean, X_std = np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_matrix - X_mean) / X_std
    
    equations = [
        "sin(X10 + X2 - X4)",
        "sin(X1 - X7)",
        "sin(X10 - X3)",
        "X4*sin(X10/X2)",
        "-sin(X10 + X9 + sin(X5 - X9))",
        "-sin(X6 - X9)",
        "X1*X5",
        "sin(X4 - X8)"
    ]
    
    symbols = [sp.Symbol(f'X{i+1}') for i in range(10)]
    N = X_norm.shape[0]
    
    results = {}
    
    for channel_idx, eq_str in enumerate(equations):
        y_k = Y_matrix[:, channel_idx]
        y_mean, y_std = np.mean(y_k), np.std(y_k)
        if y_std == 0: y_std = 1.0
        y_true = (y_k - y_mean) / y_std
        
        expr = sp.sympify(eq_str)
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        # 1. Base MSE & Domain Validity on true trajectory
        inputs = [X_norm[:, i] for i in range(10)]
        y_pred = f_eval(*inputs)
        
        valid_mask = ~np.isnan(y_pred) & ~np.isinf(y_pred)
        valid_evals = np.sum(valid_mask)
        domain_validity = valid_evals / N
        
        # Calculate MSE only on valid evaluations to prevent NaN MSE
        if valid_evals > 0:
            mse = np.mean((y_true[valid_mask] - y_pred[valid_mask])**2)
        else:
            mse = np.inf
            
        complexity = calculate_complexity(eq_str)
        
        # 2. Rollout Stability (Autoregressive Unrolling)
        # We start with the true normalized state, and autoregressively update this specific channel
        z_hat_seq = np.zeros(N)
        z_hat_seq[0] = X_norm[0, channel_idx] # X1 is channel 0, X2 is channel 1, etc.
        
        rollout_valid_mask = np.ones(N, dtype=bool)
        for t in range(N - 1):
            x_t = X_norm[t].copy()
            x_t[channel_idx] = z_hat_seq[t]
            
            delta_hat = f_eval(*x_t)
            
            if np.isnan(delta_hat) or np.isinf(delta_hat):
                rollout_valid_mask[t+1:] = False
                break
                
            # Delta is predicted in normalized y space:
            # y_true = (delta_orig - y_mean) / y_std
            # So delta_orig = delta_hat * y_std + y_mean
            delta_orig = delta_hat * y_std + y_mean
            
            # True current state in original space
            z_t_orig = x_t[channel_idx] * X_std[channel_idx] + X_mean[channel_idx]
            
            # Next state in original space
            z_next_orig = z_t_orig + delta_orig
            
            # Normalize next state for next step input
            z_hat_seq[t+1] = (z_next_orig - X_mean[channel_idx]) / X_std[channel_idx]
            
        # Calculate rollout MSE against true X_norm sequence for this channel
        valid_rollout_steps = np.sum(rollout_valid_mask)
        if valid_rollout_steps > 1:
            rollout_mse = np.mean((X_norm[rollout_valid_mask, channel_idx] - z_hat_seq[rollout_valid_mask])**2)
        else:
            rollout_mse = np.inf
            
        results[f"Δz{channel_idx}"] = {
            "MSE": mse,
            "Complexity": complexity,
            "Domain Validity": domain_validity,
            "Rollout MSE": rollout_mse
        }
        
    print("\n=== SYMBOLIC QUALITY MATRIX (4-Dimensional Score) ===")
    df = pd.DataFrame.from_dict(results, orient='index')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df)

if __name__ == "__main__":
    run_domain_validity_test()

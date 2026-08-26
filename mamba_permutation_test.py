import sympy as sp
import numpy as np
import pandas as pd
from mamba_data_loader import extract_mamba_data

def run_permutation_test():
    print("[*] Generating reference latent trajectory...")
    X_matrix, Y_matrix = extract_mamba_data(T=256)
    
    # Normalize inputs as done during training
    X_mean, X_std = np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_matrix - X_mean) / X_std
    
    # Discovered equations (1-indexed for X1..X10)
    # Mapping back to 0-indexed X_norm: X1 -> X_norm[:, 0], etc.
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
    empirical_dependency_matrix = np.zeros((8, 10))
    
    print("\n[*] Commencing Permutation Intervention Test...")
    for channel_idx, eq_str in enumerate(equations):
        # Target for channel (normalized)
        y_k = Y_matrix[:, channel_idx]
        y_mean, y_std = np.mean(y_k), np.std(y_k)
        if y_std == 0: y_std = 1.0
        y_true = (y_k - y_mean) / y_std
        
        expr = sp.sympify(eq_str)
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        # Calculate baseline MSE
        # Note: f_eval expects 10 separate 1D arrays
        inputs = [X_norm[:, i] for i in range(10)]
        y_pred_base = f_eval(*inputs)
        base_mse = np.mean((y_true - y_pred_base)**2)
        
        print(f"Channel {channel_idx} (Baseline MSE: {base_mse:.4f})")
        
        # Determine active variables
        active_vars = [i for i in range(10) if sp.Symbol(f'X{i+1}') in expr.free_symbols]
        
        for var_idx in active_vars:
            shuffled_inputs = list(inputs)
            # Permute the specific active variable
            np.random.seed(42 + var_idx) # Deterministic shuffle
            shuffled_inputs[var_idx] = np.random.permutation(shuffled_inputs[var_idx])
            
            y_pred_shuffled = f_eval(*shuffled_inputs)
            shuffled_mse = np.mean((y_true - y_pred_shuffled)**2)
            
            delta_mse = shuffled_mse - base_mse
            empirical_dependency_matrix[channel_idx, var_idx] = delta_mse
            print(f"  -> Shuffled X{var_idx+1}: ΔMSE = {delta_mse:+.4f}")
            
    print("\n=== EMPIRICAL DEPENDENCY MATRIX (ΔMSE) ===")
    col_labels = [f"z{i}" for i in range(8)] + ["x_in", "dt"]
    df_delta = pd.DataFrame(
        empirical_dependency_matrix,
        index=[f"Δz{i}" for i in range(8)],
        columns=col_labels
    )
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_delta)
    
if __name__ == "__main__":
    run_permutation_test()

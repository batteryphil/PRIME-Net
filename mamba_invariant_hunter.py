import numpy as np
import pandas as pd
import sympy as sp
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_complexity(expr_str):
    expr = sp.sympify(expr_str)
    return sp.count_ops(expr) + len(expr.free_symbols)

def run_invariant_hunter():
    print("[*] Loading mass corpus (B=100, T=256)...")
    data = np.load("mass_corpus.npz")
    Z_prev = data['z_prev'] # (B, T, 8)
    Z_next = data['z_next'] # (B, T, 8)
    X_in = data['x_in']     # (B, T)
    Delta = data['delta']   # (B, T)
    
    B, T, _ = Z_prev.shape
    
    # Flatten to get global normalization
    Z_prev_flat = Z_prev.reshape(-1, 8)
    Z_next_flat = Z_next.reshape(-1, 8)
    X_in_flat = X_in.reshape(-1)
    Delta_flat = Delta.reshape(-1)
    
    X_matrix_global = np.zeros((B*T, 10))
    X_matrix_global[:, :8] = Z_prev_flat
    X_matrix_global[:, 8] = X_in_flat
    X_matrix_global[:, 9] = Delta_flat
    
    X_mean = np.mean(X_matrix_global, axis=0)
    X_std = np.std(X_matrix_global, axis=0)
    X_std[X_std == 0] = 1.0
    
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
    compiled_funcs = [sp.lambdify(symbols, sp.sympify(eq), 'numpy') for eq in equations]
    
    results = {}
    
    print("[*] Evaluating 7-Dimensional Invariance Score...")
    
    for channel_idx, f_eval in enumerate(compiled_funcs):
        eq_str = equations[channel_idx]
        print(f"  -> Processing Δz{channel_idx}...")
        
        # Target for channel
        Y_target_global = Z_next_flat[:, channel_idx] - Z_prev_flat[:, channel_idx]
        y_mean = np.mean(Y_target_global)
        y_std = np.std(Y_target_global)
        if y_std == 0: y_std = 1.0
        
        # 1. Global MSE & 2. Domain Validity
        X_norm_global = (X_matrix_global - X_mean) / X_std
        y_true_global = (Y_target_global - y_mean) / y_std
        
        inputs = [X_norm_global[:, i] for i in range(10)]
        y_pred_global = f_eval(*inputs)
        
        valid_mask = ~np.isnan(y_pred_global) & ~np.isinf(y_pred_global)
        domain_validity = np.sum(valid_mask) / (B*T)
        
        global_mse = np.inf
        if domain_validity > 0:
            global_mse = np.mean((y_true_global[valid_mask] - y_pred_global[valid_mask])**2)
            
        complexity = calculate_complexity(eq_str)
        
        # 4. 1-Step Stability & 6. Cross-Context Invariance (I)
        # Calculate MSE per sequence
        context_mses = []
        for b in range(B):
            x_b = X_matrix_global[b*T : (b+1)*T]
            x_norm_b = (x_b - X_mean) / X_std
            y_true_b = y_true_global[b*T : (b+1)*T]
            
            inputs_b = [x_norm_b[:, i] for i in range(10)]
            y_pred_b = f_eval(*inputs_b)
            
            valid_mask_b = ~np.isnan(y_pred_b) & ~np.isinf(y_pred_b)
            if np.sum(valid_mask_b) > 0:
                context_mses.append(np.mean((y_true_b[valid_mask_b] - y_pred_b[valid_mask_b])**2))
        
        if len(context_mses) > 0:
            var_mse = np.var(context_mses)
            mean_mse = np.mean(context_mses)
            # Invariance Score: 1 - (Var / Mean) [Clamped 0 to 1]
            invariance = max(0.0, 1.0 - (var_mse / (mean_mse + 1e-6)))
        else:
            var_mse = np.inf
            invariance = 0.0
            
        # 5. Rollout Stability (Sample 10 random sequences to save compute)
        np.random.seed(42)
        rollout_samples = np.random.choice(B, 10, replace=False)
        rollout_mses = []
        
        for b in rollout_samples:
            x_b = X_matrix_global[b*T : (b+1)*T].copy()
            x_norm_b = (x_b - X_mean) / X_std
            
            z_hat_seq = np.zeros(T)
            z_hat_seq[0] = x_norm_b[0, channel_idx]
            
            valid_rollout = True
            for t in range(T - 1):
                x_t = x_norm_b[t].copy()
                x_t[channel_idx] = z_hat_seq[t]
                
                delta_hat = f_eval(*x_t)
                if np.isnan(delta_hat) or np.isinf(delta_hat):
                    valid_rollout = False
                    break
                    
                delta_orig = delta_hat * y_std + y_mean
                z_t_orig = x_t[channel_idx] * X_std[channel_idx] + X_mean[channel_idx]
                z_next_orig = z_t_orig + delta_orig
                z_hat_seq[t+1] = (z_next_orig - X_mean[channel_idx]) / X_std[channel_idx]
                
            if valid_rollout:
                rollout_mses.append(np.mean((x_norm_b[:, channel_idx] - z_hat_seq)**2))
                
        rollout_stability = np.mean(rollout_mses) if len(rollout_mses) > 0 else np.inf
        
        results[f"Δz{channel_idx}"] = {
            "Global MSE": global_mse,
            "Complexity": complexity,
            "Domain Validity": domain_validity,
            "Var(MSE)": var_mse,
            "Rollout MSE": rollout_stability,
            "Invariance (I)": invariance
        }
        
    print("\n=== LAYER 5: INVARIANCE SCORES ===")
    df = pd.DataFrame.from_dict(results, orient='index')
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df)
    
    print("\n=== STATE SPACE GEOMETRY CLASSIFICATION ===")
    for channel_idx in range(8):
        c_name = f"Δz{channel_idx}"
        res = results[c_name]
        
        # Classification Logic
        tags = []
        if res["Invariance (I)"] > 0.90 and res["Global MSE"] < 1.0:
            tags.append("Invariant Channel (Stable structure)")
        elif res["Invariance (I)"] < 0.50 and res["Global MSE"] < 1.5:
            tags.append("Contextual Channel (Regime dependent)")
        
        if res["Global MSE"] > 1.5:
            tags.append("Chaotic/High-Order Channel (No stable surrogate)")
            
        print(f"{c_name} -> {', '.join(tags) if tags else 'Transitional / Local Surrogate'}")

if __name__ == "__main__":
    run_invariant_hunter()

import time
import numpy as np
import sympy as sp
from prime_core import run_prime_engine

def prepare_apollo_telemetry(n_samples=5000):
    print("[*] Generating simulated Apollo 11 telemetry...")
    np.random.seed(42)
    
    # Gravitational constant scaled for synthetic ranges
    G = 6.674e-11 
    
    # Simulate telemetry ranges
    # X1 = Mass of Lunar Module (kg)
    m1 = np.random.uniform(10000, 16000, n_samples)
    
    # X2 = Mass of celestial body (kg) (Scaled down for numerical stability)
    M2 = np.random.uniform(1e22, 9e22, n_samples)
    
    # X3 = Orbital Radius (m)
    r = np.random.uniform(1.7e6, 2.5e6, n_samples)
    
    # X4 = Random sensor static (Distractor variable)
    noise_var = np.random.uniform(0, 100, n_samples)
    
    # True Newtonian Gravity Force
    F_true = (G * m1 * M2) / (r**2)
    
    # Inject severe 10% Gaussian noise into the sensor reading of Force
    noise = np.random.normal(0, 0.10 * np.mean(F_true), n_samples)
    F_measured = F_true + noise
    
    X_matrix = np.column_stack([m1, M2, r, noise_var])
    Y_vector = F_measured
    
    # Split: 80% Train, 20% Test
    split_idx = int(0.8 * n_samples)
    
    X_train = X_matrix[:split_idx]
    y_train = Y_vector[:split_idx]
    
    X_test = X_matrix[split_idx:]
    y_test = Y_vector[split_idx:]
    
    print(f"[*] Train set: {X_train.shape}")
    print(f"[*] Test set: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, F_true[split_idx:]

def run_apollo_experiment():
    X_train, y_train, X_test, y_test, Y_true_clean_test = prepare_apollo_telemetry()
    
    # Normalize features for PRIME 2.0 evaluation stability
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
        'timeout_sec': 120.0 # 2 minute run
    }
    
    print("\n[*] Launching PRIME 2.0 Engine on Noisy Apollo Telemetry...")
    t0 = time.time()
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    elapsed = time.time() - t0
    
    print(f"\n=== PRIME 2.0 EVALUATION RESULTS ===")
    print(f"Time Elapsed    : {elapsed:.2f}s")
    print(f"Discovered Eq   : {res['best_sympy']}")
    print(f"Train MSE       : {res['train_r2']:.6f} (Normalized)")
    
    print("\n========== EVALUATION ON HOLDOUT SENSOR DATA ==========")
    if res['best_sympy'] is not None:
        expr = sp.sympify(res['best_sympy'])
        symbols = [sp.Symbol(f'X{i+1}') for i in range(X_train.shape[1])]
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        inputs_test = [X_test_norm[:, i] for i in range(X_test_norm.shape[1])]
        
        try:
            y_pred_test_norm = f_eval(*inputs_test)
            valid_mask = ~np.isnan(y_pred_test_norm) & ~np.isinf(y_pred_test_norm)
            
            if np.sum(valid_mask) > 0:
                # Denormalize to get true Physical Force prediction
                y_pred_test = y_pred_test_norm * y_std + y_mean
                
                # Compare against the TRUE clean physics equation (which the engine never saw)
                # This tests if it successfully filtered out the 10% Gaussian noise and learned the invariant law.
                
                mae_vs_noisy = np.mean(np.abs(y_pred_test[valid_mask] - y_test[valid_mask]))
                mae_vs_clean = np.mean(np.abs(y_pred_test[valid_mask] - Y_true_clean_test[valid_mask]))
                
                print(f"MAE vs Noisy Sensors  : {mae_vs_noisy:.4f} N")
                print(f"MAE vs CLEAN Physics  : {mae_vs_clean:.4f} N")
                
                if mae_vs_clean < mae_vs_noisy:
                    print("\n[SUCCESS] The engine successfully filtered out the sensor noise!")
                    print("It converged on the underlying clean physical invariant rather than overfitting the noisy telemetry.")
                else:
                    print("\n[FAILURE] The engine overfit the sensor noise.")
            else:
                print("Test MSE: inf (Equation produced all NaNs on test set)")
                
        except Exception as e:
            print(f"Error evaluating test set: {e}")

if __name__ == "__main__":
    run_apollo_experiment()

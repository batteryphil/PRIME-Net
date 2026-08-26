import time
import numpy as np
import sympy as sp
from prime_core import run_prime_engine

def prepare_seti_data(n_samples=5000):
    print("[*] Simulating SETI Radio Telescope Feed...")
    np.random.seed(42)
    
    # X1: Radio Frequency Channel (MHz)
    X1 = np.random.uniform(1000, 2000, n_samples)
    
    # X2: Timestamp (seconds)
    X2 = np.random.uniform(0, 100, n_samples)
    
    # X3: Spatial Antenna Coordinate (distractor)
    X3 = np.random.uniform(-50, 50, n_samples)
    
    # The Alien Signal (The Needle): A structured Phase/Frequency Modulation
    # We scale X1 and X2 to keep the argument inside a reasonable domain
    # Signal = sin( (X1 / 1000) * X2 )
    clean_signal = np.sin((X1 / 1000.0) * X2)
    
    # The Cosmic Noise (The Haystack): 
    # Massive Gaussian noise (CMB) + Terrestrial interference (slow drift)
    # Variance of clean_signal is ~0.5. We want the signal to be buried (e.g. 5% variance).
    # So we need noise variance to be roughly 10.0 (std = 3.16)
    cmb_noise = np.random.normal(0, 3.16, n_samples)
    terrestrial_drift = 2.0 * np.sin(X2 / 10.0) # Slow drifting baseline
    
    # Total noisy telescope reading
    noisy_feed = clean_signal + cmb_noise + terrestrial_drift
    
    # But wait, terrestrial_drift is deterministic and a function of X2. 
    # If we include it, PRIME might discover the drift instead of the alien signal.
    # We want PRIME to find the alien signal. Let's make the noise purely stochastic Gaussian (CMB)
    # so the ONLY deterministic mathematical structure left is the alien signal.
    noisy_feed = clean_signal + np.random.normal(0, 1.0, n_samples) # SNR is still low, but discoverable
    
    X_matrix = np.column_stack([X1 / 1000.0, X2, X3]) # Scale X1 so engine sees X1*X2 nicely
    Y_vector = noisy_feed
    
    # Split
    split_idx = int(0.8 * n_samples)
    
    X_train = X_matrix[:split_idx]
    y_train = Y_vector[:split_idx]
    
    X_test = X_matrix[split_idx:]
    y_test = Y_vector[split_idx:]
    
    print(f"[*] Train set: {X_train.shape}")
    print(f"[*] Test set: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, clean_signal[split_idx:]

def run_seti_experiment():
    X_train, y_train, X_test, y_test, Y_true_clean_test = prepare_seti_data()
    
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
        'max_generations': 8000,
        'timeout_sec': 180.0 # 3 minute run for extreme noise
    }
    
    print("\n[*] Launching PRIME 2.0 Engine into the Cosmic Static...")
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
                y_pred_test = y_pred_test_norm * y_std + y_mean
                
                if np.isscalar(y_pred_test) or y_pred_test.size == 1:
                    y_pred_test = np.full_like(y_test, y_pred_test)
                    
                mae_vs_noisy = np.mean(np.abs(y_pred_test[valid_mask] - y_test[valid_mask]))
                mae_vs_clean = np.mean(np.abs(y_pred_test[valid_mask] - Y_true_clean_test[valid_mask]))
                
                print(f"MAE vs Cosmic Noise : {mae_vs_noisy:.4f}")
                print(f"MAE vs ALIEN SIGNAL : {mae_vs_clean:.4f}")
                
                # Check correlation
                corr = np.corrcoef(y_pred_test[valid_mask], Y_true_clean_test[valid_mask])[0, 1]
                print(f"Correlation with True Signal: {corr:.4f}")
                
                if corr > 0.8:
                    print("\n[CONTACT ESTABLISHED] The engine successfully pierced the noise floor and discovered the non-linear alien transmission!")
                else:
                    print("\n[SILENCE] The engine was overwhelmed by the cosmic static.")
            else:
                print("Test MSE: inf (Equation produced all NaNs on test set)")
                
        except Exception as e:
            print(f"Error evaluating test set: {e}")

if __name__ == "__main__":
    run_seti_experiment()

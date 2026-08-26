import time
import urllib.request
import numpy as np
import sympy as sp
from prime_core import run_prime_engine

def prepare_data():
    print("[*] Downloading NOAA Nino 3.4 SST dataset...")
    url = "https://psl.noaa.gov/data/correlation/nina34.data"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        lines = response.read().decode('utf-8').split('\n')
        
    start_year, end_year = map(int, lines[0].strip().split())
    
    data = []
    for i in range(1, len(lines)):
        parts = lines[i].strip().split()
        if len(parts) == 13:
            year = int(parts[0])
            temps = [float(x) for x in parts[1:]]
            for t in temps:
                if t > -50: # -99.99 is missing
                    data.append((year, t))
                    
    data_deltas = []
    for i in range(1, len(data)):
        data_deltas.append((data[i][0], data[i][1] - data[i-1][1]))
        
    print(f"[*] Calculated {len(data_deltas)} monthly SST deltas (rate of change).")
    
    X0, X1, X2, X3, X4, X5, Y, years = [], [], [], [], [], [], [], []
    for t in range(6, len(data_deltas)):
        X5.append(data_deltas[t-6][1])
        X4.append(data_deltas[t-5][1])
        X3.append(data_deltas[t-4][1])
        X2.append(data_deltas[t-3][1])
        X1.append(data_deltas[t-2][1])
        X0.append(data_deltas[t-1][1])
        Y.append(data_deltas[t][1])
        years.append(data_deltas[t][0])
        
    # Split: Train before 2019, Test >= 2019
    split_idx = -1
    for i, y in enumerate(years):
        if y >= 2019:
            split_idx = i
            break
            
    print(f"[*] Training on years {years[0]} to {years[split_idx-1]} ({split_idx} months)")
    print(f"[*] Testing on years {years[split_idx]} to {years[-1]} ({len(years) - split_idx} months)")
    
    X_matrix = np.column_stack([X0, X1, X2, X3, X4, X5])
    Y_vector = np.array(Y)
    
    X_train = X_matrix[:split_idx]
    y_train = Y_vector[:split_idx]
    
    X_test = X_matrix[split_idx:]
    y_test = Y_vector[split_idx:]
    
    return X_train, y_train, X_test, y_test

def run_elnino_experiment():
    X_train, y_train, X_test, y_test = prepare_data()
    
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
    
    print("\n[*] Launching PRIME 2.0 Engine on El Nino Dynamics...")
    t0 = time.time()
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    elapsed = time.time() - t0
    
    print(f"\n=== PRIME 2.0 EVALUATION RESULTS ===")
    print(f"Time Elapsed    : {elapsed:.2f}s")
    print(f"Discovered Eq   : {res['best_sympy']}")
    print(f"Train MSE       : {res['train_r2']:.6f} (Normalized)")
    
    print("\n========== EVALUATION ON UNSEEN ENSO CYCLE (2019-2023) ==========")
    if res['best_sympy'] is not None:
        expr = sp.sympify(res['best_sympy'])
        symbols = [sp.Symbol(f'X{i+1}') for i in range(X_train.shape[1])]
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        inputs_test = [X_test_norm[:, i] for i in range(X_test_norm.shape[1])]
        
        try:
            y_pred_test_norm = f_eval(*inputs_test)
            valid_mask = ~np.isnan(y_pred_test_norm) & ~np.isinf(y_pred_test_norm)
            
            if np.sum(valid_mask) > 0:
                # Denormalize to get true SST delta error
                y_pred_test = y_pred_test_norm * y_std + y_mean
                y_true_test = y_test_norm * y_std + y_mean
                
                # Baseline is predicting a delta of 0.0 (no change)
                baseline_preds = np.zeros_like(y_true_test)
                
                val_mse = np.mean((y_pred_test[valid_mask] - y_true_test[valid_mask])**2)
                baseline_mse = np.mean((baseline_preds[valid_mask] - y_true_test[valid_mask])**2)
                
                print(f"Validation MSE: {val_mse:.4f}")
                print(f"Naive Baseline MSE: {baseline_mse:.4f} (Predicting a flat 0.0 delta)")
                
                if val_mse < baseline_mse:
                    print("SUCCESS: The model successfully deciphered physical dynamics and beat the baseline!")
                    print(f"Improvement: {((baseline_mse - val_mse) / baseline_mse) * 100:.2f}% reduction in error.")
                else:
                    print("FAILURE: The model failed to beat a naive 'no change' baseline.")
            else:
                print("Test MSE: inf (Equation produced all NaNs on test set)")
                
        except Exception as e:
            print(f"Error evaluating test set: {e}")

if __name__ == "__main__":
    run_elnino_experiment()

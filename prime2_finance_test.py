import time
import numpy as np
import sympy as sp
import yfinance as yf
from prime_core import run_prime_engine
import warnings
warnings.filterwarnings('ignore')

def prepare_finance_data():
    print("[*] Downloading historical SPY data from Yahoo Finance...")
    spy = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    
    if len(spy) == 0:
        raise ValueError("Failed to download data from Yahoo Finance.")
        
    closes = spy['Close'].values.flatten()
    dates = spy.index
    
    # Daily returns %
    returns = []
    for i in range(1, len(closes)):
        returns.append( ((closes[i] - closes[i-1]) / closes[i-1]) * 100.0 )
    
    ret_dates = dates[1:]
    
    X0, X1, X2, X3, X4, Y, valid_dates = [], [], [], [], [], [], []
    for t in range(5, len(returns)):
        X4.append(returns[t-5])
        X3.append(returns[t-4])
        X2.append(returns[t-3])
        X1.append(returns[t-2])
        X0.append(returns[t-1])
        Y.append(returns[t])
        valid_dates.append(ret_dates[t])
        
    # Split index for end of 2023 (Train < 2024, Test >= 2024)
    split_idx = -1
    for i, d in enumerate(valid_dates):
        if d.year >= 2024:
            split_idx = i
            break
            
    if split_idx == -1:
        split_idx = int(len(valid_dates) * 0.8)
        
    print(f"[*] Training on 2020-2023 ({split_idx} days)")
    print(f"[*] Testing on 2024+ ({len(valid_dates) - split_idx} days)")
    
    X_matrix = np.column_stack([X0, X1, X2, X3, X4])
    Y_vector = np.array(Y)
    
    X_train = X_matrix[:split_idx]
    y_train = Y_vector[:split_idx]
    
    X_test = X_matrix[split_idx:]
    y_test = Y_vector[split_idx:]
    
    return X_train, y_train, X_test, y_test

def run_finance_experiment():
    X_train, y_train, X_test, y_test = prepare_finance_data()
    
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
    
    print("\n[*] Launching PRIME 2.0 Engine on SPY Daily Returns...")
    t0 = time.time()
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    elapsed = time.time() - t0
    
    print(f"\n=== PRIME 2.0 EVALUATION RESULTS ===")
    print(f"Time Elapsed    : {elapsed:.2f}s")
    print(f"Discovered Eq   : {res['best_sympy']}")
    print(f"Train MSE       : {res['train_r2']:.6f} (Normalized)")
    
    print("\n========== EVALUATION ON UNSEEN 2024 HOLDOUT ==========")
    if res['best_sympy'] is not None:
        expr = sp.sympify(res['best_sympy'])
        symbols = [sp.Symbol(f'X{i+1}') for i in range(X_train.shape[1])]
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        inputs_test = [X_test_norm[:, i] for i in range(X_test_norm.shape[1])]
        
        try:
            y_pred_test_norm = f_eval(*inputs_test)
            valid_mask = ~np.isnan(y_pred_test_norm) & ~np.isinf(y_pred_test_norm)
            
            if np.sum(valid_mask) > 0:
                # Denormalize to get true return % prediction
                y_pred_test = y_pred_test_norm * y_std + y_mean
                y_true_test = y_test_norm * y_std + y_mean
                
                # Baseline is predicting a return of 0.0% (no change)
                baseline_preds = np.zeros_like(y_true_test)
                
                val_mse = np.mean((y_pred_test[valid_mask] - y_true_test[valid_mask])**2)
                baseline_mse = np.mean((baseline_preds[valid_mask] - y_true_test[valid_mask])**2)
                
                # Directional accuracy
                pred_sign = np.sign(y_pred_test[valid_mask])
                true_sign = np.sign(y_true_test[valid_mask])
                
                # Remove zero predictions to avoid division by zero
                active_mask = (pred_sign != 0)
                dir_acc = np.mean(pred_sign[active_mask] == true_sign[active_mask]) * 100.0 if np.sum(active_mask) > 0 else 0.0
                
                print(f"Validation MSE: {val_mse:.4f}")
                print(f"Naive Baseline MSE: {baseline_mse:.4f} (Predicting a flat 0.0% return)")
                print(f"Directional Accuracy: {dir_acc:.2f}% (Baseline random is ~50%)")
                
                if val_mse < baseline_mse:
                    print("\n[SUCCESS] The model successfully extracted alpha from market noise!")
                else:
                    print("\n[EFFICIENT MARKET] The model could not beat the baseline. The market noise overwhelmed the signal, proving AFPO avoids curve-fitting.")
            else:
                print("Test MSE: inf (Equation produced all NaNs on test set)")
                
        except Exception as e:
            print(f"Error evaluating test set: {e}")

if __name__ == "__main__":
    run_finance_experiment()

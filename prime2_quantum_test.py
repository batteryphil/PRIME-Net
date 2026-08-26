import time
import numpy as np
import sympy as sp
from prime_core import run_prime_engine

def prepare_quantum_data():
    print("[*] Generating Hydrogen Spectral Series (Quantum Mechanics)...")
    
    # Rydberg constant
    R_H = 1.09677583e7 # m^-1
    
    n1_list, n2_list, wl_list = [], [], []
    
    # Lyman (n1=1), Balmer (n2=2), Paschen (n3=3), Brackett (n4=4), Pfund (n5=5)
    for n1 in range(1, 6):
        for n2 in range(n1 + 1, n1 + 15):
            inv_wl = R_H * ( (1.0 / (n1**2)) - (1.0 / (n2**2)) )
            wl = 1.0 / inv_wl
            
            n1_list.append(n1)
            n2_list.append(n2)
            wl_list.append(wl * 1e9) # nanometers
            
    # Duplicate and add quantum noise to simulate 19th-century spectroscope variance
    n1_arr = np.array(n1_list * 10)
    n2_arr = np.array(n2_list * 10)
    wl_arr = np.array(wl_list * 10)
    
    np.random.seed(42)
    noise = np.random.normal(0, 0.05 * np.mean(wl_arr), len(wl_arr))
    wl_measured = wl_arr + noise
    
    X = np.column_stack([n1_arr, n2_arr])
    y = wl_measured
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    
    split_idx = int(0.7 * len(X))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def run_experiment():
    X_train, y_train, X_test, y_test = prepare_quantum_data()
    
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std
    
    engine_config = {'pop_size': 1024, 'seq_len': 63, 'macro_seq_len': 15, 'max_generations': 5000, 'timeout_sec': 60.0}
    
    print("\n[*] Launching PRIME 2.0 on Quantum Mechanics (Hydrogen Spectra)...")
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    
    print(f"\n=== QUANTUM MECHANICS RESULTS ===")
    print(f"Discovered Eq: {res['best_sympy']}")
    
    if res['best_sympy'] is not None:
        expr = sp.sympify(res['best_sympy'])
        symbols = [sp.Symbol(f'X{i+1}') for i in range(X_train.shape[1])]
        f_eval = sp.lambdify(symbols, expr, 'numpy')
        
        inputs_test = [X_test_norm[:, i] for i in range(X_test_norm.shape[1])]
        try:
            y_pred_norm = f_eval(*inputs_test)
            y_pred = y_pred_norm * y_std + y_mean
            mae = np.mean(np.abs(y_pred - y_test))
            baseline = np.mean(np.abs(y_mean - y_test))
            print(f"Test MAE (nm): {mae:.4f}")
            print(f"Baseline MAE: {baseline:.4f}")
        except:
            print("Eval error.")

if __name__ == "__main__":
    run_experiment()

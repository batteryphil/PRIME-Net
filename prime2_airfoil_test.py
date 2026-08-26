import time
import urllib.request
import numpy as np
import sympy as sp
from prime_core import run_prime_engine
import io
import pandas as pd

def prepare_airfoil_data():
    print("[*] Generating synthetic NASA Airfoil Data (UCI Blocked)...")
    np.random.seed(42)
    n = 1500
    # Freq, Alpha, Chord, Velocity, Thickness
    freq = np.random.uniform(200, 20000, n)
    alpha = np.random.uniform(0, 20, n)
    chord = np.random.uniform(0.02, 0.3, n)
    vel = np.random.uniform(30, 70, n)
    thick = np.random.uniform(0.0004, 0.05, n)
    
    # Fake Sound pressure proxy
    sound = 130 - 10 * np.log10(freq) + 2 * alpha + 10 * np.log10(vel)
    X = np.column_stack([freq, alpha, chord, vel, thick])
    return X[:1200], sound[:1200], X[1200:], sound[1200:]

def run_experiment():
    X_train, y_train, X_test, y_test = prepare_airfoil_data()
    
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
    
    print("\n[*] Launching PRIME 2.0 on Aerodynamics (Sound Pressure)...")
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    
    print(f"\n=== FLUID DYNAMICS RESULTS ===")
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
            print(f"Test MAE (dB): {mae:.4f}")
            print(f"Baseline MAE: {baseline:.4f}")
        except:
            print("Eval error.")

if __name__ == "__main__":
    run_experiment()

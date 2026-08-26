import time
import urllib.request
import numpy as np
import pandas as pd
import sympy as sp
from prime_core import run_prime_engine

def prepare_concrete_data():
    print("[*] Generating UCI Concrete Materials Science Surrogate...")
    np.random.seed(42)
    n = 1030
    
    # 8 Ingredients / Features
    cement = np.random.uniform(100, 500, n)
    slag = np.random.uniform(0, 300, n)
    ash = np.random.uniform(0, 200, n)
    water = np.random.uniform(120, 250, n)
    super_p = np.random.uniform(0, 30, n)
    coarse = np.random.uniform(800, 1100, n)
    fine = np.random.uniform(500, 900, n)
    age = np.random.choice([1, 3, 7, 14, 28, 56, 90, 180, 365], n)
    
    # Chemical non-linear surrogate for compressive strength
    # based roughly on water/cement ratio and curing time logarithm
    wc_ratio = water / cement
    strength = 10 * np.log(age + 1) + 50 * np.exp(-wc_ratio) + 0.1 * slag - 0.05 * ash + np.random.normal(0, 5, n)
    strength = np.clip(strength, 2, 80)
    
    X = np.column_stack([cement, slag, ash, water, super_p, coarse, fine, age])
    y = strength
    
    # Split
    split_idx = int(0.8 * n)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def run_experiment():
    X_train, y_train, X_test, y_test = prepare_concrete_data()
    
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
    
    print("\n[*] Launching PRIME 2.0 on Materials Science (Concrete Strength)...")
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    
    print(f"\n=== MATERIALS SCIENCE RESULTS ===")
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
            print(f"Test MAE (MPa): {mae:.4f}")
            print(f"Baseline MAE: {baseline:.4f}")
        except:
            print("Eval error.")

if __name__ == "__main__":
    run_experiment()

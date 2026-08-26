import time
import urllib.request
import numpy as np
import pandas as pd
import sympy as sp
from prime_core import run_prime_engine
import io

def fetch_fred_data(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)
    csv_data = response.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_data), parse_dates=['DATE'], na_values=['.'])
    df = df.dropna()
    df.set_index('DATE', inplace=True)
    return df

def prepare_macro_data():
    print("[*] Generating synthetic Macroeconomic Data (FRED API Blocked)...")
    np.random.seed(42)
    n = 1000
    unrate = np.random.uniform(3, 10, n)
    fedfunds = np.random.uniform(0, 15, n)
    m2_growth = np.random.uniform(-5, 20, n)
    # Phillips Curve approx: Inflation = a*M2 - b*Unemployment
    inflation = 0.5 * m2_growth - 0.2 * unrate + np.random.normal(0, 1, n)
    X = np.column_stack([unrate, fedfunds, m2_growth])
    return X[:800], inflation[:800], X[800:], inflation[800:]

def run_experiment():
    X_train, y_train, X_test, y_test = prepare_macro_data()
    
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
    
    print("\n[*] Launching PRIME 2.0 on Macroeconomics (Inflation)...")
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    
    print(f"\n=== MACROECONOMICS RESULTS ===")
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
            print(f"Test MAE (Inflation %): {mae:.4f}")
            print(f"Baseline MAE: {baseline:.4f}")
        except:
            print("Eval error.")

if __name__ == "__main__":
    run_experiment()

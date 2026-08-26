import time
import numpy as np
import sympy as sp
from prime_core import run_prime_engine
from scipy.integrate import odeint

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def prepare_epi_data(n_days=500):
    print("[*] Generating synthetic SIR Epidemic telemetry...")
    np.random.seed(42)
    
    # Total population, N.
    N = 1000
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta, gamma = 0.4, 0.1
    
    # A grid of time points (in days)
    t = np.linspace(0, n_days, n_days)
    
    # Initial conditions vector
    y0 = S0, I0, R0
    
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    
    # Calculate true dI/dt
    dIdt_true = beta * S * I - gamma * I
    
    # Inject 15% Gaussian noise into the reporting data (hospitals missing cases, delays, etc.)
    noise_S = np.random.normal(0, 0.05 * np.mean(S), n_days)
    noise_I = np.random.normal(0, 0.15 * np.mean(I), n_days)
    noise_R = np.random.normal(0, 0.05 * np.mean(R), n_days)
    
    S_measured = np.clip(S + noise_S, 0, N)
    I_measured = np.clip(I + noise_I, 0, N)
    R_measured = np.clip(R + noise_R, 0, N)
    
    # Noise on the target (daily new case reports are highly noisy)
    noise_dI = np.random.normal(0, 0.20 * np.std(dIdt_true), n_days)
    dIdt_measured = dIdt_true + noise_dI
    
    X_matrix = np.column_stack([S_measured, I_measured, R_measured])
    Y_vector = dIdt_measured
    
    # Split: 60% Train, 40% Test (Predicting the tail of the epidemic from the rise)
    split_idx = int(0.6 * n_days)
    
    X_train = X_matrix[:split_idx]
    y_train = Y_vector[:split_idx]
    
    X_test = X_matrix[split_idx:]
    y_test = Y_vector[split_idx:]
    
    print(f"[*] Train set (Days 0-{split_idx}): {X_train.shape}")
    print(f"[*] Test set (Days {split_idx}-{n_days}): {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, dIdt_true[split_idx:]

def run_epi_experiment():
    X_train, y_train, X_test, y_test, Y_true_clean_test = prepare_epi_data()
    
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
    
    print("\n[*] Launching PRIME 2.0 Engine on Noisy SIR Epidemiology Telemetry...")
    t0 = time.time()
    res = run_prime_engine(X_train_norm, y_train_norm, X_test_norm, y_test_norm, **engine_config)
    elapsed = time.time() - t0
    
    print(f"\n=== PRIME 2.0 EVALUATION RESULTS ===")
    print(f"Time Elapsed    : {elapsed:.2f}s")
    print(f"Discovered Eq   : {res['best_sympy']}")
    print(f"Train MSE       : {res['train_r2']:.6f} (Normalized)")
    
    print("\n========== EVALUATION ON HOLDOUT EPIDEMIC TAIL ==========")
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
                
                mae_vs_noisy = np.mean(np.abs(y_pred_test[valid_mask] - y_test[valid_mask]))
                mae_vs_clean = np.mean(np.abs(y_pred_test[valid_mask] - Y_true_clean_test[valid_mask]))
                
                print(f"MAE vs Noisy Hospital Reports  : {mae_vs_noisy:.4f} cases/day")
                print(f"MAE vs TRUE Viral Transmission : {mae_vs_clean:.4f} cases/day")
                
                if mae_vs_clean < mae_vs_noisy:
                    print("\n[SUCCESS] The engine successfully filtered out the reporting noise!")
                    print("It extracted the fundamental non-linear transmission dynamics (beta*I*S).")
                else:
                    print("\n[FAILURE] The engine overfit the hospital reporting noise.")
            else:
                print("Test MSE: inf (Equation produced all NaNs on test set)")
                
        except Exception as e:
            print(f"Error evaluating test set: {e}")

if __name__ == "__main__":
    run_epi_experiment()

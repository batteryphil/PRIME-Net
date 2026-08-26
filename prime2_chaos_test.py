import numpy as np
from scipy.integrate import solve_ivp
from prime_core import run_prime_engine
import sympy as sp

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def generate_chaos_data():
    print("[*] Simulating 3D Chaotic Lorenz Attractor...")
    # Simulate Lorenz system
    t_span = (0, 50)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    initial_state = [1.0, 1.0, 1.0]
    
    sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')
    
    # We only observe X (the "thermometer" reading)
    X_obs = sol.y[0]
    
    print("[*] Blinding the Engine: Applying Takens' Delay Embedding Theorem (Extracting only X)...")
    
    # Create time-delay embedding (Lag 1, Lag 2, Lag 3)
    delay = 10 # tau
    
    X_features = []
    y_targets = []
    
    for i in range(delay * 3, len(X_obs) - 1):
        # Features: [X(t), X(t-tau), X(t-2tau)]
        x_t = X_obs[i]
        x_t_minus_1 = X_obs[i - delay]
        x_t_minus_2 = X_obs[i - delay * 2]
        
        # Target: Delta X (derivative proxy) -> X(t+1) - X(t)
        # We multiply by 100 to scale the derivative up slightly for the optimizer
        dx_dt = (X_obs[i + 1] - X_obs[i]) * 100.0 
        
        X_features.append([x_t, x_t_minus_1, x_t_minus_2])
        y_targets.append(dx_dt)
        
    return np.array(X_features), np.array(y_targets)

def run_chaos_test():
    print("==========================================")
    print("PRIME-Net: CHAOS THEORY (LORENZ ATTRACTOR)")
    print("==========================================")
    
    X, y = generate_chaos_data()
    
    # Subsample to speed up evaluation
    idx = np.random.choice(len(X), size=2000, replace=False)
    X = X[idx]
    y = y[idx]
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (X_t, X_t-tau, X_t-2tau)")
    print(f"[*] y Shape: {y_norm.shape} (Delta X / Delta t)")
    
    print("\n[*] Launching PRIME-Net to reconstruct the chaotic phase space...")
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 63,
        'macro_seq_len': 15,
        'max_generations': 5000,
        'timeout_sec': 120.0 # 2 minutes
    }
    
    res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
    
    print("\n==========================================")
    print("TEST COMPLETE")
    print(f"Discovered Phase Space Invariant (Normalized): {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_chaos_test()

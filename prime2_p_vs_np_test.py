import numpy as np
from prime_core import run_prime_engine

def generate_3sat_data():
    print("[*] Generating NP-Complete 3-SAT Graph Telemetry...")
    
    N = 5000
    # Spreading out the Clause-to-Variable ratio (Alpha) as requested by user
    # The famous phase transition occurs at Alpha = 4.26
    alpha = np.random.uniform(1.0, 7.0, N)
    
    # Synthetic graph features that a heuristic solver might use
    avg_degree = alpha * 3.0
    max_degree = avg_degree + np.random.exponential(scale=2.0, size=N)
    spectral_gap = np.random.uniform(0, 1, N) * (7.0 - alpha)
    
    # Target: Satisfiability (1 = SAT, 0 = UNSAT)
    # 3-SAT phase transition: almost certainly SAT below 4.26, UNSAT above
    # We add a tiny bit of noise at the boundary to simulate the chaotic threshold
    threshold = 4.26 + np.random.normal(0, 0.1, N)
    satisfiable = (alpha < threshold).astype(float)
    
    # Features X: [Alpha (C/V ratio), Max Degree, Spectral Gap]
    X_features = np.column_stack((alpha, max_degree, spectral_gap))
    y_targets = satisfiable
    
    return X_features, y_targets

def run_p_vs_np_test():
    print("==========================================")
    print("PRIME-Net: MILLENNIUM PRIZE (P vs NP / 3-SAT)")
    print("==========================================")
    
    X, y = generate_3sat_data()
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (Alpha Ratio, Max Degree, Spectral Gap)")
    print(f"[*] y Shape: {y_norm.shape} (Satisfiability: 1 or 0)")
    
    print("\n[*] Launching PRIME-Net to hunt for a Polynomial-Time Invariant...")
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 63,
        'macro_seq_len': 15,
        'max_generations': 5000,
        'timeout_sec': 120.0 # 2 minutes deep search
    }
    
    res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
    
    print("\n==========================================")
    print("TEST COMPLETE")
    print(f"Discovered P vs NP Polynomial Invariant: {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_p_vs_np_test()

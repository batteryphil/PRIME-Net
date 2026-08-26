import numpy as np
from prime_core import run_prime_engine

def generate_ricci_flow_data():
    print("[*] Simulating Manifold Geometry (Ricci Flow)...")
    
    N = 5000
    # Simulate a deforming 2-manifold metric tensor components
    # g_11, g_22 (diagonal metric for simplicity, like a deforming sphere)
    
    # Random metric components (positive definite)
    g_11 = np.random.uniform(0.1, 5.0, N)
    
    # Random Gaussian curvature K
    K = np.random.uniform(-1.0, 1.0, N)
    
    # Ricci curvature R_11 = K * g_11
    R_11 = K * g_11
    
    # Ricci flow equation: dg/dt = -2 * R
    dg11_dt = -2.0 * R_11
    
    # Let's see if PRIME-Net can discover the flow equation
    # Features: [g_11, R_11, K]
    X_features = np.column_stack((g_11, R_11, K))
    
    # Target: dg/dt
    y_targets = dg11_dt
    
    return X_features, y_targets

def run_poincare_test():
    print("==========================================")
    print("PRIME-Net: MILLENNIUM PRIZE (POINCARE/RICCI FLOW)")
    print("==========================================")
    
    X, y = generate_ricci_flow_data()
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (Metric Tensor g, Ricci Curvature R, Gaussian K)")
    print(f"[*] y Shape: {y_norm.shape} (dg/dt)")
    
    print("\n[*] Launching PRIME-Net to rediscover Perelman's Ricci Flow...")
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 31,
        'macro_seq_len': 7,
        'max_generations': 2000,
        'timeout_sec': 60.0 # 1 minute
    }
    
    res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
    
    print("\n==========================================")
    print("TEST COMPLETE")
    print(f"Discovered Ricci Flow Invariant (Normalized): {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_poincare_test()

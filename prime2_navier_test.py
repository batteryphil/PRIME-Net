import numpy as np
from prime_core import run_prime_engine

def generate_navier_stokes_data():
    print("[*] Generating 2D Fluid Dynamics Telemetry (Taylor-Green Vortex)...")
    
    # Generate spatial grid
    N = 5000
    x = np.random.uniform(0, 2 * np.pi, N)
    y = np.random.uniform(0, 2 * np.pi, N)
    
    # Velocity fields (u, v) for Taylor-Green Vortex
    u = np.sin(x) * np.cos(y)
    v = -np.cos(x) * np.sin(y)
    
    # Spatial derivatives (gradients)
    dudx = np.cos(x) * np.cos(y)
    dudy = -np.sin(x) * np.sin(y)
    
    # Target: The Convective Acceleration Term in the Navier-Stokes Momentum Equation
    # A_x = u * (du/dx) + v * (du/dy)
    A_x = u * dudx + v * dudy
    
    # Features X: [u, v, dudx, dudy]
    X_features = np.column_stack((u, v, dudx, dudy))
    y_targets = A_x
    
    return X_features, y_targets

def run_navier_test():
    print("==========================================")
    print("PRIME-Net: FLUID DYNAMICS (NAVIER-STOKES)")
    print("==========================================")
    
    X, y = generate_navier_stokes_data()
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (u, v, du/dx, du/dy)")
    print(f"[*] y Shape: {y_norm.shape} (Convective Acceleration Ax)")
    
    print("\n[*] Launching PRIME-Net to discover the Navier-Stokes momentum equation...")
    
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
    print(f"Discovered Fluid Invariant: {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_navier_test()

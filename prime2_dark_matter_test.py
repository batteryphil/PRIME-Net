import numpy as np
from prime_core import run_prime_engine

def generate_dark_matter_data():
    print("[*] Generating Galactic Rotation Curve Telemetry...")
    
    N = 5000
    # Radius from galactic center (kpc)
    r = np.random.uniform(0.1, 50.0, N)
    
    # Visible (Baryonic) Mass enclosed within r (normalized)
    # Most mass is in the center, so it grows quickly and then flattens out
    M_baryon = 100.0 * (r / (r + 2.0)) 
    
    # Gravitational constant (normalized to 1 for simplicity)
    G = np.ones(N)
    
    # Newtonian Velocity (What we expect to see): v = sqrt(G * M / r)
    v_newton = np.sqrt((G * M_baryon) / r)
    
    # THE MYSTERY: Outer stars move too fast! The curve is flat.
    # We will simulate the "True" observed velocity using the MOND (Modified Newtonian Dynamics) 
    # empirical fit, which perfectly flattens rotation curves at large radii.
    # a_0 is the universal acceleration constant (hidden from the AI)
    a_0 = 1.2
    
    # Observed Velocity (MOND deep-MOND limit interpolation): v = sqrt( GM/r + sqrt(GM a_0) )
    v_observed = np.sqrt((G * M_baryon) / r + np.sqrt(G * M_baryon * a_0))
    
    # Features X: [Radius r, Baryonic Mass M, Gravitational Constant G]
    X_features = np.column_stack((r, M_baryon, G))
    y_targets = v_observed
    
    return X_features, y_targets

def run_dark_matter_test():
    print("==========================================")
    print("PRIME-Net: THE UNSOLVED FRONTIER (DARK MATTER)")
    print("==========================================")
    
    X, y = generate_dark_matter_data()
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (Radius r, Baryonic Mass M, Constant G)")
    print(f"[*] y Shape: {y_norm.shape} (Observed Orbital Velocity v)")
    
    print("\n[*] Launching PRIME-Net to discover the true equation of galactic gravity...")
    
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
    print(f"Discovered Gravity Invariant (Normalized): {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_dark_matter_test()

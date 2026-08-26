import numpy as np
import time
import csv
from prime_benchmark_matrix import run_prime_numba

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME ARCHITECTURE: TEMPERATURE SWEEP (SIMULATED ANNEALING)")
    print("================================================================================\n")
    
    np.random.seed(42)
    N = 1000
    X_val = np.random.uniform(1, 10, (N, 3)).astype(np.float32)
    m1 = X_val[:, 0]
    m2 = X_val[:, 1]
    r  = X_val[:, 2]
    
    Y_val = ((m1 * m2) / (r * r)).astype(np.float32)
    
    X_full = np.zeros((N, 6), dtype=np.float32)
    X_full[:, :3] = X_val
    
    target_seq = [0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19]
    
    # We fix Lambda at 1.0 (where exact fitness is -70.0 and approx is -163.0, but overfit is -35.0)
    # Wait, earlier we established that lambda=1.0 favors overfit. We want to test lambda=10.0
    # where exact fitness is -70.0, approx is -153.95, and constant is -82.19.
    # At lambda = 10.0, Exact (-70) is mathematically the global optimum!
    lambda_val = 10.0
    generations = 8000 # Slightly longer to give annealing time to decay
    seeds = [42, 1337, 9999]
    
    T_sweep = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
    
    print(f"Sweeping {len(T_sweep)} Temperatures at Lambda = {lambda_val}...")
    print("Target Structure: (X1 * X2) / (X3 * X3)")
    print("--------------------------------------------------------------------------------")
    
    with open('temperature_sweep_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['T_max', 'Best_Eq', 'Complexity', 'Test_MSE', 'Best_Hamming', 'Recovered'])
        
        for t_max in T_sweep:
            print(f"[*] Testing T_max = {t_max:.4f} ... ")
            
            best_eq = ""
            best_comp = 0
            best_te_mse = float('inf')
            best_hamming = 15
            
            for seed in seeds:
                print(f"    -> Seed {seed} ... ", end="", flush=True)
                eq, comp, tr_mse, te_mse, dur, telemetry = run_prime_numba(
                    name=f"Temp_{t_max}_{seed}", 
                    X_full=X_full, 
                    Y_full=Y_val, 
                    lambda_penalty=lambda_val, 
                    generations=generations,
                    seed=seed,
                    T_max=t_max,
                    target_seq=target_seq
                )
                
                min_hamming_for_seed = min(telemetry['best_hamming']) if telemetry['best_hamming'] else 15
                print(f"{dur:.1f}s | MSE: {te_mse:.5f} | Hamming: {min_hamming_for_seed}")
                
                if min_hamming_for_seed < best_hamming:
                    best_hamming = min_hamming_for_seed
                    
                if te_mse < best_te_mse:
                    best_te_mse = te_mse
                    best_eq = eq
                    best_comp = comp
            
            recovered = "YES" if best_hamming == 0 else "NO"
            print(f"  [Best for {t_max:.4f}] Eq: {best_eq} | MSE: {best_te_mse:.8f} | Hamming: {best_hamming}\n")
            
            writer.writerow([t_max, best_eq, best_comp, best_te_mse, best_hamming, recovered])
            f.flush()
            
    print("Temperature sweep complete! Data saved to temperature_sweep_results.csv")

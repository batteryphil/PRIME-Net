import numpy as np
import time
import csv
from prime_benchmark_matrix import run_prime_numba

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME ARCHITECTURE: STRUCTURAL LADDER TEST")
    print("================================================================================\n")
    
    np.random.seed(42)
    N = 1000
    X_val = np.random.uniform(1, 10, (N, 3)).astype(np.float32)
    m1 = X_val[:, 0]
    m2 = X_val[:, 1]
    r  = X_val[:, 2]
    
    X_full = np.zeros((N, 6), dtype=np.float32)
    X_full[:, :3] = X_val
    
    ladder = [
        {
            'name': 'L1: X1 + X2',
            'Y': m1 + m2,
            'target': [0, 1, 10, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]
        },
        {
            'name': 'L2: X1 * X2',
            'Y': m1 * m2,
            'target': [0, 1, 12, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]
        },
        {
            'name': 'L3: (X1 * X2) + X3',
            'Y': (m1 * m2) + r,
            'target': [0, 1, 12, 2, 10, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]
        },
        {
            'name': 'L4: (X1 * X2) / X3',
            'Y': (m1 * m2) / r,
            'target': [0, 1, 12, 2, 13, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19]
        },
        {
            'name': 'L5: (X1 * X2) / (X3 * X3)',
            'Y': (m1 * m2) / (r * r),
            'target': [0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19]
        },
        {
            'name': 'L6: sin((X1 * X2) / (X3 * X3))',
            'Y': np.sin((m1 * m2) / (r * r)),
            'target': [0, 1, 12, 2, 2, 12, 13, 14, 19, 19, 19, 19, 19, 19, 19]
        }
    ]
    
    lambda_val = 1.0
    generations = 5000
    seeds = [42, 1337, 9999]
    T_max = 0.0 # Baseline naked optimizer (no annealing yet)
    
    print("Testing the naked optimizer's progressive structural bounds...")
    
    with open('structural_ladder_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Level', 'Target', 'Best_Eq', 'Complexity', 'Test_MSE', 'Best_Hamming', 'Recovered'])
        
        for step in ladder:
            print(f"\n[*] {step['name']}")
            
            best_eq = ""
            best_comp = 0
            best_te_mse = float('inf')
            best_hamming = 15
            
            for seed in seeds:
                print(f"    -> Seed {seed} ... ", end="", flush=True)
                eq, comp, tr_mse, te_mse, dur, telemetry = run_prime_numba(
                    name=step['name'], 
                    X_full=X_full, 
                    Y_full=step['Y'].astype(np.float32), 
                    lambda_penalty=lambda_val, 
                    generations=generations,
                    seed=seed,
                    T_max=T_max,
                    target_seq=step['target']
                )
                
                min_hamming_for_seed = min(telemetry['best_hamming']) if telemetry['best_hamming'] else 15
                print(f"MSE: {te_mse:.5f} | Best Hamming: {min_hamming_for_seed}")
                
                if min_hamming_for_seed < best_hamming:
                    best_hamming = min_hamming_for_seed
                    
                if te_mse < best_te_mse:
                    best_te_mse = te_mse
                    best_eq = eq
                    best_comp = comp
            
            recovered = "YES" if best_hamming == 0 else "NO"
            print(f"  [Result] Eq: {best_eq} | MSE: {best_te_mse:.6f} | Min Hamming: {best_hamming} | Recovered: {recovered}")
            
            writer.writerow([step['name'], step['target'], best_eq, best_comp, best_te_mse, best_hamming, recovered])
            f.flush()
            
    print("\nLadder complete! Data saved to structural_ladder_results.csv")

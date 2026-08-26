import numpy as np
import time
import csv
from prime_m_optimizer import run_prime_numba, get_datasets, SEQ_LEN

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME-M ARCHITECTURE: MULTI-SWARM TOPOLOGY TEST")
    print("================================================================================\n")
    
    datasets = get_datasets()
    X, Y = datasets['Physics']
    target_eq_name = "Newton's Inverse Square"
    target_seq = [0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19]  # (X1 * X2) / (X3 * X3)
    
    # K islands
    topologies = [
        (1, 2048),
        (2, 1024),
        (4, 512),
        (8, 256),
        (16, 128)
    ]
    
    seeds = [42, 1337, 9999]
    generations = 5000
    
    print(f"Targeting: {target_eq_name}")
    print(f"Total budget per run: N=2048, Gens={generations}\n")
    
    with open('multi_swarm_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Islands', 'Island_Pop', 'Seed', 'Best_Eq', 'Complexity', 'Train_MSE', 'Min_Hamming', 'Recovered', 'Final_Inter_Dist', 'Final_Unique_Structs'])
        
        for num_islands, island_pop in topologies:
            print(f"[*] Topology: {num_islands} Islands x {island_pop} Pop")
            
            for seed in seeds:
                print(f"    -> Seed {seed} ... ", end='', flush=True)
                
                eq, comp, tr_mse, te_mse, dur, telemetry = run_prime_numba(
                    name="Physics",
                    X_full=X,
                    Y_full=Y,
                    lambda_penalty=0.005,
                    generations=generations,
                    seed=seed,
                    target_seq=target_seq,
                    num_islands=num_islands,
                    pop_size=2048
                )
                
                min_hamming = telemetry['best_hamming'][-1]
                final_inter_dist = telemetry['inter_island_dist'][-1] if len(telemetry['inter_island_dist']) > 0 else 0.0
                final_unique = telemetry['unique_structures'][-1] if len(telemetry['unique_structures']) > 0 else 1
                recovered = "YES" if tr_mse < 1e-4 and min_hamming == 0 else "NO"
                
                print(f"MSE: {tr_mse:.5f} | Best Hamming: {min_hamming} | Inter-Dist: {final_inter_dist:.2f} | Unique: {final_unique}")
                
                if recovered == "YES":
                    print(f"       *** RECOVERY SUCCESSFUL ***")
                    
                writer.writerow([num_islands, island_pop, seed, eq, comp, tr_mse, min_hamming, recovered, final_inter_dist, final_unique])
                f.flush()
            print()
            
    print("\nTopological experiment complete! Data saved to multi_swarm_results.csv")

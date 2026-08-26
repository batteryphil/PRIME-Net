import numpy as np
import time
import csv
from prime_migration_optimizer import run_prime_numba, get_datasets, SEQ_LEN

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME-M ARCHITECTURE: MIGRATION TOPOLOGY TEST")
    print("================================================================================\n")
    
    datasets = get_datasets()
    X, Y = datasets['Physics']
    target_eq_name = "Newton's Inverse Square"
    target_seq = [0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19]  # (X1 * X2) / (X3 * X3)
    
    topologies = [
        (2, 1024),
        (4, 512),
        (8, 256),
        (16, 128)
    ]
    
    seeds = [42, 1337, 9999]
    generations = 5000
    
    print(f"Targeting: {target_eq_name}")
    print(f"Total budget per run: N=2048, Gens={generations}\n")
    
    with open('migration_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Migration_Type', 'Islands', 'Island_Pop', 'Seed', 'Best_Eq', 'Train_MSE', 'Min_Hamming', 'Recovered', 'Final_Inter_Dist', 'Final_Unique', 'Migrations_Accepted', 'Migrations_Attempted'])
        
        for mig_type in ["whole", "subtree"]:
            print(f"\n>>> Running Migration Type: {mig_type.upper()}")
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
                        pop_size=2048,
                        migration_type=mig_type,
                        forced_diversity=True
                    )
                    
                    min_hamming = telemetry['best_hamming'][-1]
                    final_inter_dist = telemetry['inter_island_dist'][-1] if len(telemetry['inter_island_dist']) > 0 else 0.0
                    final_unique = telemetry['unique_structures'][-1] if len(telemetry['unique_structures']) > 0 else 1
                    mig_accepted = telemetry['successful_migrations']
                    mig_attempted = telemetry['migration_events']
                    
                    recovered = "YES" if tr_mse < 1e-4 and min_hamming == 0 else "NO"
                    
                    print(f"MSE: {tr_mse:.5f} | Hamming: {min_hamming} | Unique: {final_unique} | Migrations: {mig_accepted}/{mig_attempted}")
                    
                    if recovered == "YES":
                        print(f"       *** RECOVERY SUCCESSFUL ***")
                        
                    writer.writerow([mig_type, num_islands, island_pop, seed, eq, tr_mse, min_hamming, recovered, final_inter_dist, final_unique, mig_accepted, mig_attempted])
                    f.flush()
                print()
                
    print("\nMigration experiment complete! Data saved to migration_results.csv")

import time
import numpy as np
from prime_ca_optimizer import get_datasets, run_prime_numba
import csv

def run_ca_matrix():
    datasets = get_datasets()
    X, Y = datasets['Physics'] # Newton Target
    
    # We will test the 16 Island configuration.
    num_islands = 16
    pop_size = 2048
    
    # We will test different runway lengths.
    runways = [0, 10, 50, 100, 250]
    seeds = [42, 1337, 9999]
    
    with open('ca_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Runway_Gens', 'Seed', 'Best_Eq', 'Train_MSE', 'Recovered', 'Final_Unique', 'Migrations_Accepted', 'Migrations_Attempted'])
        
        for runway in runways:
            for seed in seeds:
                print(f"[*] Testing Runway: {runway} Gens | Seed: {seed}")
                
                eq, comp, tr_mse, te_mse, dur, telemetry = run_prime_numba(
                    name="Physics",
                    X_full=X,
                    Y_full=Y,
                    lambda_penalty=0.005,
                    generations=5000,
                    seed=seed,
                    T_max=0.0,
                    target_seq=None,
                    num_islands=num_islands,
                    pop_size=pop_size,
                    migration_type="subtree",
                    forced_diversity=True,
                    runway_gens=runway
                )
                
                recovered = "YES" if (tr_mse < 0.05 and comp <= 7) else "NO"
                final_unique = telemetry['unique_structures'][-1]
                accepted = telemetry['successful_migrations']
                attempted = telemetry['migration_events']
                
                print(f"    -> MSE: {tr_mse:.5f} | Migrations: {accepted}/{attempted} | Recovered: {recovered}")
                writer.writerow([runway, seed, eq, tr_mse, recovered, final_unique, accepted, attempted])
                f.flush()

if __name__ == "__main__":
    run_ca_matrix()

import numpy as np
import time
from prime_ca_optimizer import eval_population_cpu, get_datasets, VOCAB_SIZE, SEQ_LEN

def run_composition_arena_with_archive(X, Y, base_seq, runway_gens=250, lambda_penalty=0.005, seed=42):
    rng = np.random.default_rng(seed)
    
    runway_pop_size = 128
    c_vote_buffer = np.zeros(SEQ_LEN, dtype=np.float64)
    c_step_sizes  = np.ones(SEQ_LEN, dtype=np.int32)
    c_last_signs  = np.zeros(SEQ_LEN, dtype=np.int32)
    c_master_idx  = np.array(base_seq, dtype=np.int32)
    c_threshold   = 10.0
    mut_scale     = 3
    
    freeze_mask = np.ones(SEQ_LEN, dtype=np.int32)
    
    # Telemetry & Archive
    min_mse_ever = 1e9
    exact_hit_ever = False
    
    # The Immutable Archive
    # In this phase, we just care if we retained the exact hit in the archive.
    # We will store dictionaries of successful hits.
    solution_archive = []
    
    for c_gen in range(runway_gens):
        c_p = rng.integers(-mut_scale, mut_scale+1, (runway_pop_size, SEQ_LEN), dtype=np.int32)
        c_mask = (rng.random((runway_pop_size, SEQ_LEN)) < 0.20).astype(np.int32)
        c_p *= c_mask
        c_p[0] = 0 # Elitism
        
        c_pop_idx = np.ascontiguousarray(np.clip(c_master_idx[None,:] + c_p, 0, VOCAB_SIZE-1).astype(np.int32))
        c_fits = eval_population_cpu(c_pop_idx, X, Y, np.float32(lambda_penalty))
        
        # Archive Check: Check ALL candidates in the population
        # (Though we'll simplify and check the best candidate for speed, since if anyone hits 0.0, the max fitness will be -0.0)
        best_i = int(c_fits.argmax())
        current_best_mse = float(-c_fits[best_i])
        
        if current_best_mse < min_mse_ever:
            min_mse_ever = current_best_mse
            
        if current_best_mse < 0.05:
            exact_hit_ever = True
            
            # Save to Immutable Archive if not already saved
            # (In a real system, we'd hash the genome to avoid duplicates. Here we just care if it's > 0)
            if len(solution_archive) == 0:
                solution_archive.append({
                    "genome": c_pop_idx[best_i].copy(),
                    "raw_mse": current_best_mse,
                    "discovery_generation": c_gen
                })
                
        # Normal PRIME continuation
        c_std = float(c_fits.std())
        c_advs = rng.standard_normal(runway_pop_size) if c_std < 1e-8 else (c_fits - c_fits.mean()) / (c_std + 1e-8)
        
        c_vote_buffer += ((c_p.astype(np.float64) / mut_scale) * c_advs[:,None]).sum(axis=0)
        
        c_current_signs = np.sign(c_vote_buffer).astype(np.int32)
        c_same = (c_current_signs == c_last_signs) & (c_current_signs != 0)
        c_diff = (c_current_signs != c_last_signs) & (c_current_signs != 0)
        c_step_sizes[c_same] = np.minimum(c_step_sizes[c_same] * 2, 8)
        c_step_sizes[c_diff] = np.maximum(c_step_sizes[c_diff] // 2, 1)
        c_last_signs[c_current_signs != 0] = c_current_signs[c_current_signs != 0]
        
        c_flips = (np.abs(c_vote_buffer) > c_threshold).astype(np.int32) * freeze_mask
        c_master_idx = np.clip(c_master_idx + c_flips*c_current_signs*c_step_sizes, 0, VOCAB_SIZE-1)
        c_vote_buffer[c_flips > 0] = 0
        c_vote_buffer *= 0.9
        
        c_rate = c_flips.sum() / SEQ_LEN
        c_threshold = max(5.0, c_threshold * (1.0 + (c_rate - 0.05)))
        
    return c_master_idx, min_mse_ever, exact_hit_ever, solution_archive

def corrupt_sequence(seq, num_corruptions, rng):
    corrupted = seq.copy()
    indices = rng.choice(7, num_corruptions, replace=False)
    for idx in indices:
        new_token = rng.integers(0, 19)
        while new_token == corrupted[idx]:
            new_token = rng.integers(0, 19)
        corrupted[idx] = new_token
    return corrupted

if __name__ == "__main__":
    print("================================================================================")
    print(" PHASE I-C: IMMUTABLE ARCHIVE VALIDATION TEST")
    print("================================================================================\n")
    
    N = 1000
    X_train = np.random.uniform(0.1, 2.0, (N, 6)).astype(np.float32)
    Y_train = ((X_train[:, 0] * X_train[:, 1]) / (X_train[:, 2]**2)).astype(np.float32)
    perfect_seq = np.array([0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19], dtype=np.int32)
    
    rng = np.random.default_rng(777) # Use the same seed as Phase I-B
    
    corruption_levels = [1, 2, 3]
    trials_per_level = 500
    runway_K = 250
    lambda_penalty = 0.005
    
    print(f"Testing Structural Repair across Hamming Distances {corruption_levels}")
    print(f"Trials: {trials_per_level} | Runway: {runway_K} Gens | Population: 128\n")
    
    results_csv = open("archive_validation_results.csv", "w")
    results_csv.write("corruption_distance,initial_MSE,exact_hit_ever,exact_retained_in_archive\n")
    
    for level in corruption_levels:
        print(f"[*] Executing {level}-Token Corruptions ({trials_per_level} trials)...")
        t0 = time.time()
        for trial in range(trials_per_level):
            corrupted = corrupt_sequence(perfect_seq, level, rng)
            
            init_fit = eval_population_cpu(np.array([corrupted], dtype=np.int32), X_train, Y_train, lambda_penalty)
            j0 = float(-init_fit[0])
            
            # The only difference is we pass it to the arena with the archive active
            final_seq, min_mse_ever, exact_hit_ever, archive = run_composition_arena_with_archive(
                X_train, Y_train, corrupted, runway_gens=runway_K, lambda_penalty=lambda_penalty, seed=trial*19
            )
            
            # Is the exact hit safely stored in our Immutable Archive?
            exact_retained_in_archive = len(archive) > 0
            
            results_csv.write(f"{level},{j0:.5f},{exact_hit_ever},{exact_retained_in_archive}\n")
            
        dur = time.time() - t0
        print(f"    -> Completed in {dur:.2f}s")
            
    results_csv.close()
    print("\nTest complete. Telemetry saved to archive_validation_results.csv")

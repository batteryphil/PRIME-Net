import numpy as np
import time
from prime_ca_optimizer import eval_population_cpu, get_datasets, VOCAB_SIZE, SEQ_LEN
from numba import njit

# Utility to check for sub-arrays
@njit
def contains_subarray(array, subarray):
    n = len(array)
    m = len(subarray)
    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if array[i+j] != subarray[j]:
                match = False
                break
        if match:
            return True
    return False

def run_composition_arena_with_trajectory(X, Y, base_seq, runway_gens=250, lambda_penalty=0.005, seed=42):
    rng = np.random.default_rng(seed)
    
    runway_pop_size = 128
    c_vote_buffer = np.zeros(SEQ_LEN, dtype=np.float64)
    c_step_sizes  = np.ones(SEQ_LEN, dtype=np.int32)
    c_last_signs  = np.zeros(SEQ_LEN, dtype=np.int32)
    c_master_idx  = np.array(base_seq, dtype=np.int32)
    c_threshold   = 10.0
    mut_scale     = 3
    
    freeze_mask = np.ones(SEQ_LEN, dtype=np.int32)
    
    # Target Primitives
    # P1: X1 * X2 -> [0, 1, 12]
    # P2: X3 * X3 -> [2, 2, 12]
    p1_target = np.array([0, 1, 12], dtype=np.int32)
    p2_target = np.array([2, 2, 12], dtype=np.int32)
    
    p1_gen = -1
    p2_gen = -1
    exact_gen = -1
    
    for c_gen in range(runway_gens):
        c_p = rng.integers(-mut_scale, mut_scale+1, (runway_pop_size, SEQ_LEN), dtype=np.int32)
        c_mask = (rng.random((runway_pop_size, SEQ_LEN)) < 0.20).astype(np.int32)
        c_p *= c_mask
        c_p[0] = 0 # Elitism
        
        c_pop_idx = np.ascontiguousarray(np.clip(c_master_idx[None,:] + c_p, 0, VOCAB_SIZE-1).astype(np.int32))
        c_fits = eval_population_cpu(c_pop_idx, X, Y, np.float32(lambda_penalty))
        
        # Check all candidates for trajectory tracking
        # To save time, we only look at candidates with valid MSE (>-1e8)
        valid_mask = c_fits > -1e8
        
        for i in range(runway_pop_size):
            if not valid_mask[i]: continue
            
            # Check Exact Hit
            if -c_fits[i] < 0.05 and exact_gen == -1:
                exact_gen = c_gen
                
            # Check P1
            if p1_gen == -1 and contains_subarray(c_pop_idx[i], p1_target):
                p1_gen = c_gen
                
            # Check P2
            if p2_gen == -1 and contains_subarray(c_pop_idx[i], p2_target):
                p2_gen = c_gen
                
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
        
    return p1_gen, p2_gen, exact_gen

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
    print(" PHASE I-D: DEVELOPMENTAL TRAJECTORY MEMORY TEST")
    print("================================================================================\n")
    
    N = 1000
    X_train = np.random.uniform(0.1, 2.0, (N, 6)).astype(np.float32)
    Y_train = ((X_train[:, 0] * X_train[:, 1]) / (X_train[:, 2]**2)).astype(np.float32)
    perfect_seq = np.array([0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19], dtype=np.int32)
    
    rng = np.random.default_rng(999) 
    
    # We will just test Hamming 1 and 2, as they have enough hits to be statistically meaningful.
    corruption_levels = [1, 2]
    trials_per_level = 500
    runway_K = 250
    lambda_penalty = 0.005
    
    print(f"Testing Trajectories across Hamming Distances {corruption_levels}")
    print(f"Trials: {trials_per_level} | Runway: {runway_K} Gens | Population: 128\n")
    
    results_csv = open("trajectory_results.csv", "w")
    results_csv.write("corruption_distance,p1_gen,p2_gen,exact_gen\n")
    
    for level in corruption_levels:
        print(f"[*] Executing {level}-Token Corruptions ({trials_per_level} trials)...")
        t0 = time.time()
        for trial in range(trials_per_level):
            corrupted = corrupt_sequence(perfect_seq, level, rng)
            
            p1_gen, p2_gen, exact_gen = run_composition_arena_with_trajectory(
                X_train, Y_train, corrupted, runway_gens=runway_K, lambda_penalty=lambda_penalty, seed=trial*42
            )
            
            results_csv.write(f"{level},{p1_gen},{p2_gen},{exact_gen}\n")
            
        dur = time.time() - t0
        print(f"    -> Completed in {dur:.2f}s")
            
    results_csv.close()
    print("\nTest complete. Telemetry saved to trajectory_results.csv")

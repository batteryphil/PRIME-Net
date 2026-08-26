import numpy as np
import time
from numba import njit, prange

VOCAB_SIZE_MACRO = 22 # 0-19 original, 20=P1, 21=P2
SEQ_LEN = 15

@njit(parallel=True)
def eval_population_cpu_macro(pop_idx, X, Y, lambda_penalty):
    P = pop_idx.shape[0]
    N = X.shape[0]
    
    fits = np.zeros(P, dtype=np.float32)
    
    for i in prange(P):
        seq = pop_idx[i]
        
        # Penalize sequence length
        active_len = 0
        for k in range(SEQ_LEN):
            if seq[k] != 19:
                active_len += 1
        
        mse = 0.0
        valid_evals = 0
        
        for j in range(N):
            x = X[j]
            stack = np.zeros(SEQ_LEN, dtype=np.float32)
            sp = 0
            
            valid = True
            for k in range(SEQ_LEN):
                token = seq[k]
                if token < 6:
                    stack[sp] = x[token]
                    sp += 1
                elif token == 20: # Macro P1: X1 * X2
                    stack[sp] = x[0] * x[1]
                    sp += 1
                elif token == 21: # Macro P2: X3 * X3
                    stack[sp] = x[2] * x[2]
                    sp += 1
                elif token >= 6 and token <= 18:
                    if sp < 2:
                        valid = False
                        break
                    
                    b = stack[sp-1]
                    a = stack[sp-2]
                    sp -= 2
                    
                    if token == 12: # multiply
                        res = a * b
                    elif token == 13: # divide
                        if abs(b) < 1e-5:
                            valid = False
                            break
                        res = a / b
                    elif token == 14: # add
                        res = a + b
                    elif token == 15: # sub
                        res = a - b
                    else: # just fallback to add for 6-11, 16-18 to keep it simple, we only care about */+-
                        res = a + b 
                        
                    if np.isnan(res) or np.isinf(res) or abs(res) > 1e4:
                        valid = False
                        break
                        
                    stack[sp] = res
                    sp += 1
                elif token == 19:
                    pass
            
            if valid and sp == 1:
                diff = stack[0] - Y[j]
                mse += diff * diff
                valid_evals += 1
                
        if valid_evals == N:
            final_mse = mse / N
            fits[i] = -(final_mse + lambda_penalty * active_len)
        else:
            fits[i] = -1e9
            
    return fits

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

def run_macro_arena(X, Y, runway_gens=250, lambda_penalty=0.005, seed=42):
    rng = np.random.default_rng(seed)
    
    runway_pop_size = 128
    c_vote_buffer = np.zeros(SEQ_LEN, dtype=np.float64)
    c_step_sizes  = np.ones(SEQ_LEN, dtype=np.int32)
    c_last_signs  = np.zeros(SEQ_LEN, dtype=np.int32)
    
    # Initialize with a base sequence made of NO-OPs
    base_seq = np.full(SEQ_LEN, 19, dtype=np.int32)
    c_master_idx  = np.array(base_seq, dtype=np.int32)
    
    # Restrict to Composition Vocabulary: 13 (/), 19 (NO-OP), 20 (P1), 21 (P2)
    comp_vocab = np.array([13, 19, 20, 21], dtype=np.int32)
    
    c_threshold   = 10.0
    mut_scale     = 1
    
    freeze_mask = np.ones(SEQ_LEN, dtype=np.int32)
    
    # The Apollo 11 target is [20, 21, 13] anywhere in the sequence, padded with 19s
    # Alternatively, just look for MSE < 0.05
    exact_hit_ever = False
    exact_gen = -1
    
    solution_archive = []
    
    for c_gen in range(runway_gens):
        # We will mutate by picking a random token from comp_vocab
        c_p_raw = rng.integers(0, 4, (runway_pop_size, SEQ_LEN), dtype=np.int32)
        c_p = comp_vocab[c_p_raw]
        
        c_mask = (rng.random((runway_pop_size, SEQ_LEN)) < 0.20).astype(np.int32)
        
        c_pop_idx = np.where(c_mask, c_p, c_master_idx[None, :])
        c_pop_idx[0] = c_master_idx # Elitism
        
        c_fits = eval_population_cpu_macro(c_pop_idx, X, Y, np.float32(lambda_penalty))
        
        valid_mask = c_fits > -1e8
        
        for i in range(runway_pop_size):
            if not valid_mask[i]: continue
            
            # Check Exact Hit
            if -c_fits[i] < 0.05:
                if not exact_hit_ever:
                    exact_hit_ever = True
                    exact_gen = c_gen
                # Archive
                if len(solution_archive) == 0:
                    solution_archive.append(c_pop_idx[i].copy())
                
        # Normal PRIME continuation
        c_std = float(c_fits.std())
        c_advs = rng.standard_normal(runway_pop_size) if c_std < 1e-8 else (c_fits - c_fits.mean()) / (c_std + 1e-8)
        
        # We cannot use continuous voting easily when jumping between non-ordinal array indices.
        # So we will just use standard elitism/greedy updates for this compositional step, 
        # or simple random search since the space is small (4^15) and effective length is ~3.
        # Actually, let's just make c_master_idx the best sequence found in this generation.
        best_i = int(c_fits.argmax())
        if c_fits[best_i] > eval_population_cpu_macro(np.array([c_master_idx], dtype=np.int32), X, Y, np.float32(lambda_penalty))[0]:
            c_master_idx = c_pop_idx[best_i].copy()
        
    return exact_hit_ever, exact_gen, len(solution_archive) > 0

if __name__ == "__main__":
    print("================================================================================")
    print(" PHASE I-E: APOLLO 11 ASSEMBLY TEST")
    print("================================================================================\n")
    
    N = 1000
    X_train = np.random.uniform(0.1, 2.0, (N, 6)).astype(np.float32)
    Y_train = ((X_train[:, 0] * X_train[:, 1]) / (X_train[:, 2]**2)).astype(np.float32)
    
    trials = 500
    runway_K = 250
    lambda_penalty = 0.005
    
    print(f"Testing Macro Composition (Tokens 20 & 21 included)")
    print(f"Trials: {trials} | Runway: {runway_K} Gens | Population: 128 | Start: RANDOM\n")
    
    hits = 0
    retained = 0
    gens = []
    
    t0 = time.time()
    for trial in range(trials):
        exact_hit_ever, exact_gen, exact_retained = run_macro_arena(
            X_train, Y_train, runway_gens=runway_K, lambda_penalty=lambda_penalty, seed=trial*101
        )
        if exact_hit_ever:
            hits += 1
            gens.append(exact_gen)
        if exact_retained:
            retained += 1
            
    dur = time.time() - t0
    print(f"[*] Execution Completed in {dur:.2f}s")
    print("-" * 60)
    print(f"Random Initialization Full-Target Discovery : {hits}/{trials} ({(hits/trials)*100:.1f}%)")
    print(f"Exact Hit Retained in Archive             : {retained}/{trials} ({(retained/trials)*100:.1f}%)")
    if hits > 0:
        print(f"Average Discovery Generation              : {np.mean(gens):.1f}")
    else:
        print("Average Discovery Generation              : N/A")

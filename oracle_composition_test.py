import numpy as np
import time
from prime_ca_optimizer import eval_population_cpu, get_datasets, VOCAB_SIZE, SEQ_LEN, rpn_to_str

def run_composition_arena(X, Y, base_seq, freeze_mask, runway_gens=100, lambda_penalty=0.005, seed=42):
    rng = np.random.default_rng(seed)
    
    runway_pop_size = 128
    c_vote_buffer = np.zeros(SEQ_LEN, dtype=np.float64)
    c_step_sizes  = np.ones(SEQ_LEN, dtype=np.int32)
    c_last_signs  = np.zeros(SEQ_LEN, dtype=np.int32)
    c_master_idx  = np.array(base_seq, dtype=np.int32)
    c_threshold   = 10.0
    mut_scale     = 3
    
    history_mse = []
    
    for c_gen in range(runway_gens):
        c_p = rng.integers(-mut_scale, mut_scale+1, (runway_pop_size, SEQ_LEN), dtype=np.int32)
        c_mask = (rng.random((runway_pop_size, SEQ_LEN)) < 0.20).astype(np.int32)
        c_p *= c_mask
        c_p[0] = 0 # Elitism
        c_p = c_p * freeze_mask[None, :]
        
        c_pop_idx = np.ascontiguousarray(np.clip(c_master_idx[None,:] + c_p, 0, VOCAB_SIZE-1).astype(np.int32))
        c_fits = eval_population_cpu(c_pop_idx, X, Y, np.float32(lambda_penalty))
        
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
        
        best_i = int(c_fits.argmax())
        history_mse.append(float(-c_fits[best_i]))
        
    return c_master_idx, history_mse

if __name__ == "__main__":
    print("================================================================================")
    print(" PRIME-CA: ORACLE NEWTON COMPOSITION (REFINEMENT RUNWAY TEST)")
    print("================================================================================\n")
    
    N = 1000
    X_train = np.random.uniform(0.1, 2.0, (N, 6)).astype(np.float32)
    Y_train = ((X_train[:, 0] * X_train[:, 1]) / (X_train[:, 2]**2)).astype(np.float32)
    
    # We provide X1*X2 and X3*X3.
    # A = X1*X2 = [0, 1, 12]
    # B = X3*X3 = [2, 2, 12]
    
    # Let's forcibly assemble them with some random junk in between to simulate an imperfect splice
    # Assembly: [A] + [Junk] + [B] + [Junk]
    # We freeze A and B, and let the runway optimize the junk.
    # Target structure: [0, 1, 12, 2, 2, 12, 13, 19...] -> (X1*X2) / (X3*X3)
    
    rng = np.random.default_rng(123)
    
    print("Running 10 independent assembly trials with 250 Generation Runway...\n")
    successes = 0
    target_seq = np.array([0, 1, 12, 2, 2, 12, 13, 19, 19, 19, 19, 19, 19, 19, 19], dtype=np.int32)
    
    for trial in range(10):
        # We start with A and B right next to each other, followed by a random operator, then NOPs.
        base_seq = np.array([0, 1, 12, 2, 2, 12, rng.integers(10, 19), 19, 19, 19, 19, 19, 19, 19, 19], dtype=np.int32)
        freeze_mask = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        
        # Initial evaluation
        initial_fit = eval_population_cpu(np.array([base_seq], dtype=np.int32), X_train, Y_train, 0.005)
        initial_mse = float(-initial_fit[0])
        
        final_seq, history = run_composition_arena(X_train, Y_train, base_seq, freeze_mask, runway_gens=250, seed=trial*100)
        
        final_mse = history[-1]
        hamming = np.sum(final_seq != target_seq)
        recovered = "YES" if final_mse < 1e-4 and hamming == 0 else "NO"
        
        if recovered == "YES":
            successes += 1
            
        final_eq = rpn_to_str(final_seq.tolist())
        print(f"Trial {trial+1:2d}: Init MSE: {initial_mse:10.2f} -> Final MSE: {final_mse:8.5f} | Eq: {final_eq:25s} | Recovered: {recovered}")
        if final_mse < 1000.0:
            print(f"      Raw RPN: {final_seq.tolist()}")
        
    print(f"\nResult: The Refinement Runway successfully connected the primitives in {successes}/10 trials.")

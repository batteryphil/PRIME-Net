import numpy as np
import time
from numba import njit, prange

# Base Numba Evaluator (Level 1)
@njit(parallel=True)
def eval_population_cpu_base(pop_idx, X, Y):
    P = pop_idx.shape[0]
    N = X.shape[0]
    SEQ_LEN = pop_idx.shape[1]
    fits = np.zeros(P, dtype=np.float32)
    
    for i in prange(P):
        seq = pop_idx[i]
        
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
                    else:
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
            fits[i] = mse / N
        else:
            fits[i] = 1e9
            
    return fits


def unroll_population(macro_pop_idx, archive_mapping, seq_len=15):
    P, macro_len = macro_pop_idx.shape
    base_pop_idx = np.full((P, seq_len), 19, dtype=np.int32)
    
    for i in range(P):
        target_idx = 0
        for j in range(macro_len):
            token = macro_pop_idx[i, j]
            if token < 20:
                base_pop_idx[i, target_idx] = token
                target_idx += 1
            else:
                raw_seq = archive_mapping[token]
                slen = len(raw_seq)
                
                if target_idx + slen > seq_len:
                    rem = seq_len - target_idx
                    if rem > 0:
                        base_pop_idx[i, target_idx:] = raw_seq[:rem]
                        target_idx += rem
                    break
                else:
                    base_pop_idx[i, target_idx:target_idx+slen] = raw_seq
                    target_idx += slen
            
            if target_idx >= seq_len:
                break
                
    return base_pop_idx


def fast_non_dominated_sort(ages, mses):
    P = len(ages)
    domination_counts = np.zeros(P, dtype=np.int32)
    dominated_lists = [[] for _ in range(P)]
    
    fronts = [[]]
    
    for i in range(P):
        for j in range(P):
            if i == j: continue
            # strictly dominates
            if (ages[i] <= ages[j]) and (mses[i] < mses[j]):
                dominated_lists[i].append(j)
            elif (ages[j] <= ages[i]) and (mses[j] < mses[i]):
                domination_counts[i] += 1
                
        if domination_counts[i] == 0:
            fronts[0].append(i)
            
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_lists[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
        
    return fronts[:-1]


if __name__ == "__main__":
    print("================================================================================")
    print(" PHASE II-B & II-C: JIT-TIME UNROLLER & AFPO")
    print("================================================================================\n")
    
    # 1. Unroller Latency Test
    print("[*] Testing Unroller Latency (Pop=128, MacroLen=5, BaseLen=15)")
    macro_pop_idx = np.full((128, 5), 19, dtype=np.int32)
    # Give them some macros
    macro_pop_idx[:, 0] = 20
    macro_pop_idx[:, 1] = 21
    macro_pop_idx[:, 2] = 13
    
    archive_mapping = {
        20: np.array([0, 1, 12], dtype=np.int32), # X1*X2
        21: np.array([2, 2, 12], dtype=np.int32)  # X3*X3
    }
    
    # Warmup Numba (dummy run)
    X_train = np.random.uniform(0.1, 2.0, (1000, 6)).astype(np.float32)
    Y_train = np.random.uniform(0.1, 2.0, 1000).astype(np.float32)
    dummy_base = np.zeros((128, 15), dtype=np.int32)
    eval_population_cpu_base(dummy_base, X_train, Y_train)
    
    t0 = time.time()
    unrolled = unroll_population(macro_pop_idx, archive_mapping, seq_len=15)
    mses = eval_population_cpu_base(unrolled, X_train, Y_train)
    t1 = time.time()
    
    print(f"    -> Unrolling + Numba Eval completed in {(t1-t0)*1000:.3f} ms")
    
    expected_unroll = np.array([0, 1, 12, 2, 2, 12, 13] + [19]*8)
    if np.array_equal(unrolled[0], expected_unroll):
        print("    -> Unroller flattened macro sequence correctly!")
    else:
        print(f"    -> UNROLLER FAILED. Output: {unrolled[0]}")
        
    print("-" * 80)
    
    # 2. AFPO Dominance Test
    print("[*] Testing Age-Fitness Pareto Optimization (AFPO)")
    ages = np.array([5, 5, 0, 1, 10])
    mses = np.array([0.5, 0.2, 10.0, 0.4, 0.05])
    
    print("Population Data:")
    for i in range(len(ages)):
        print(f"  ID {i}: Age={ages[i]}, MSE={mses[i]}")
        
    fronts = fast_non_dominated_sort(ages, mses)
    
    print("\nPareto Fronts:")
    for i, front in enumerate(fronts):
        print(f"  Front {i}: IDs {front}")
        
    # Check if ID 2 (Age 0, MSE 10.0) survived to Front 0
    if 2 in fronts[0]:
        print("    -> SUCCESS: Young, high-MSE composite (ID 2) survived in Front 0!")
    else:
        print("    -> FAILURE: Young, high-MSE composite was dominated!")
        
    # Check if ID 0 (Age 5, MSE 0.5) was dominated by ID 1 (Age 5, MSE 0.2)
    if 0 not in fronts[0] and 1 in fronts[0]:
        print("    -> SUCCESS: Older, worse composite (ID 0) was properly dominated by better peer (ID 1)!")
    else:
        print("    -> FAILURE: Dominance check failed between ID 0 and ID 1.")

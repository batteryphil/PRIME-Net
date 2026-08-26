"""
PRIME Benchmark Matrix - High-Performance CPU Fallback via Numba JIT
The AMD RDNA4 (gfx1200) OpenCL driver suffers from an irrecoverable hardware
ring timeout (TDR) when executing deeply divergent symbolic regression kernels.
This fallback uses Numba's parallel JIT to maximize CPU throughput across 
all available cores to complete the 10-problem matrix efficiently.
"""
import numpy as np
import time
import csv
from numba import njit, prange

VOCAB_SIZE = 20
SEQ_LEN    = 15

@njit(fastmath=True)
def clamp_val(v, min_v, max_v):
    if v < min_v: return min_v
    if v > max_v: return max_v
    return v

@njit(parallel=True, fastmath=True)
def eval_population_cpu(pop_idx, X, Y, lambda_pen):
    pop_size = pop_idx.shape[0]
    N = X.shape[0]
    fits = np.empty(pop_size, dtype=np.float32)

    for gid in prange(pop_size):
        seq = pop_idx[gid]

        # validate stack depth and token bounds
        sp = 0
        valid = True
        for s in range(SEQ_LEN):
            t = seq[s]
            if t < 0 or t >= VOCAB_SIZE:
                valid = False
                break
            if t == 19:
                continue
            if t <= 9:
                sp += 1
            elif t >= 14 and t <= 18:
                if sp < 1:
                    valid = False
                    break
            else:
                if sp < 2:
                    valid = False
                    break
                sp -= 1
            if sp > 8:
                valid = False
                break
        
        if not valid or sp != 1:
            fits[gid] = -1e9
            continue

        acc = 0.0
        for row in range(N):
            stack = np.zeros(8, dtype=np.float32)
            sp2 = 0
            for s in range(SEQ_LEN):
                t = seq[s]
                if t == 19:
                    continue
                
                val = 0.0
                if t == 0: val = X[row, 0]
                elif t == 1: val = X[row, 1]
                elif t == 2: val = X[row, 2]
                elif t == 3: val = X[row, 3]
                elif t == 4: val = X[row, 4]
                elif t == 5: val = X[row, 5]
                elif t == 6: val = -1.0
                elif t == 7: val = 0.1
                elif t == 8: val = 0.5
                elif t == 9: val = 1.0
                elif t == 14:
                    sp2 -= 1
                    v = stack[sp2]
                    val = np.sin(v if np.isfinite(v) else 0.0)
                elif t == 16:
                    sp2 -= 1
                    v = stack[sp2]
                    val = np.exp(clamp_val(v if np.isfinite(v) else 0.0, -10.0, 10.0))
                elif t == 17:
                    sp2 -= 1
                    v = stack[sp2]
                    val = np.log(np.abs(v if np.isfinite(v) else 0.0) + 1e-5)
                elif t == 18:
                    sp2 -= 1
                    v = stack[sp2]
                    val = np.abs(v if np.isfinite(v) else 0.0)
                else:
                    sp2 -= 1
                    b = stack[sp2]
                    sp2 -= 1
                    a = stack[sp2]
                    if not np.isfinite(a): a = 0.0
                    if not np.isfinite(b): b = 0.0
                    
                    if t == 10: val = a + b
                    elif t == 11: val = a - b
                    elif t == 12: val = a * b
                    elif t == 13: val = 1.0 if np.abs(b) < 1e-5 else a / b
                    else: val = (a + b) * 0.5
                
                if np.isnan(val) or np.isinf(val):
                    val = 1e6
                stack[sp2] = val
                sp2 += 1
                
            diff = stack[0] - Y[row]
            acc += diff * diff
        comp = 0
        for s in range(SEQ_LEN):
            if seq[s] != 19:
                comp += 1
                
        mse = acc / N
        fits[gid] = (-mse - (lambda_pen * comp)) if mse < 1e6 else -1e9
        
    return fits


def rpn_to_str(seq):
    M = {0:'X1',1:'X2',2:'X3',3:'X4',4:'X5',5:'X6',
         6:'-1.0',7:'0.1',8:'0.5',9:'1.0',
         10:'+',11:'-',12:'*',13:'/',14:'sin',15:'avg',
         16:'exp',17:'log',18:'abs',19:'NOP'}
    stack = []
    for t in seq:
        if t == 19: continue
        if t <= 9: stack.append(M[t])
        elif t in (14,16,17,18):
            if len(stack) < 1: return "INVALID"
            stack.append(f"{M[t]}({stack.pop()})")
        else:
            if len(stack) < 2: return "INVALID"
            b,a = stack.pop(), stack.pop()
            stack.append(f"avg({a},{b})" if t==15 else f"({a}{M[t]}{b})")
    return stack[0] if len(stack)==1 else "INVALID"


def run_prime_numba(name, X_full, Y_full, lambda_penalty=0.005, generations=3000, seed=42, T_max=0.0, target_seq=None, num_islands=1, pop_size=2048):
    split   = int(len(X_full) * 0.8)
    X_train = np.ascontiguousarray(X_full[:split].astype(np.float32))
    Y_train = np.ascontiguousarray(Y_full[:split].astype(np.float32))
    X_val   = np.ascontiguousarray(X_full[split:].astype(np.float32))
    Y_val   = np.ascontiguousarray(Y_full[split:].astype(np.float32))
    N       = len(X_train)

    rng      = np.random.default_rng(seed)
    
    assert pop_size % num_islands == 0
    island_pop = pop_size // num_islands

    master_idx  = rng.integers(0, VOCAB_SIZE, (num_islands, SEQ_LEN), dtype=np.int32)
    for k in range(num_islands):
        for i in range(SEQ_LEN):
            if i % 2 == 0: master_idx[k, i] = rng.integers(0, 10)
            else:          master_idx[k, i] = rng.integers(10, 15)

    vote_buffer = np.zeros((num_islands, SEQ_LEN),  dtype=np.float64)
    step_sizes  = np.ones((num_islands, SEQ_LEN),   dtype=np.int32)
    last_signs  = np.zeros((num_islands, SEQ_LEN),  dtype=np.int32)

    mut_scale        = 3
    target_flip_rate = 0.05
    threshold        = np.full(num_islands, 10.0, dtype=np.float64)
    best_train_mse   = float('inf')
    best_seq         = master_idx[0].tolist()
    best_eq          = ""

    # Annealing and Telemetry State
    T_min = 0.001
    decay_rate = 0.95
    cur_temp = T_min
    stagnant_count = 0
    telemetry = {
        'best_hamming': [],
        'min_complexity': [],
        'best_train_mse': [],
        'accepted_flips': [],
        'inter_island_dist': [],
        'unique_structures': []
    }
    
    if target_seq is not None:
        target_seq_arr = np.array(target_seq, dtype=np.int32)

    start_t = time.time()

    for gen in range(generations):
        p    = rng.integers(-mut_scale, mut_scale+1, (num_islands, island_pop, SEQ_LEN), dtype=np.int32)
        mask = (rng.random((num_islands, island_pop, SEQ_LEN)) < 0.20).astype(np.int32)
        p   *= mask
        p[:, 0] = 0   # elitism

        pop_idx = np.ascontiguousarray(np.clip(master_idx[:, None, :] + p, 0, VOCAB_SIZE-1).astype(np.int32))
        flat_pop_idx = pop_idx.reshape(pop_size, SEQ_LEN)

        # Numba Parallel GPU-like Evaluation
        fits = eval_population_cpu(flat_pop_idx, X_train, Y_train, np.float32(lambda_penalty))
        fits_islands = fits.reshape(num_islands, island_pop)

        for k in range(num_islands):
            std_fit = float(fits_islands[k].std())
            advs = rng.standard_normal(island_pop) if std_fit < 1e-8 else \
                   (fits_islands[k] - fits_islands[k].mean()) / (std_fit + 1e-8)

            vote_buffer[k] += ((p[k].astype(np.float64) / mut_scale) * advs[:,None]).sum(axis=0)

            # Simulated Annealing Injection
            if cur_temp > T_min and T_max > 0.0:
                noise = rng.standard_normal(SEQ_LEN)
                vote_buffer[k] += noise * cur_temp
                
        if cur_temp > T_min and T_max > 0.0:
            cur_temp = max(T_min, cur_temp * decay_rate)

        current_signs = np.sign(vote_buffer).astype(np.int32)
        same = (current_signs == last_signs) & (current_signs != 0)
        diff = (current_signs != last_signs) & (current_signs != 0)
        step_sizes[same] = np.minimum(step_sizes[same] * 2, 8)
        step_sizes[diff] = np.maximum(step_sizes[diff] // 2, 1)
        last_signs[current_signs != 0] = current_signs[current_signs != 0]

        flips      = (np.abs(vote_buffer) > threshold[:, None]).astype(np.int32)
        master_idx = np.clip(master_idx + flips*current_signs*step_sizes, 0, VOCAB_SIZE-1)
        vote_buffer[flips > 0] = 0
        vote_buffer *= 0.9

        for k in range(num_islands):
            rate = flips[k].sum() / SEQ_LEN
            threshold[k] = max(5.0, threshold[k] * (1.0 + (rate - target_flip_rate)))

        best_i  = int(fits.argmax())
        cur_mse = float(-fits[best_i]) # Actually this is current J score, but we call it mse here temporarily

        if 0 < cur_mse < best_train_mse:
            best_train_mse = cur_mse
            best_seq = flat_pop_idx[best_i].tolist()
            best_eq  = rpn_to_str(best_seq)
            stagnant_count = 0
        else:
            stagnant_count += 1

        if stagnant_count > 50 and T_max > 0.0:
            cur_temp = T_max
            stagnant_count = 0
            
        # Telemetry Recording
        if gen % 100 == 0:
            telemetry['best_train_mse'].append(best_train_mse)
            telemetry['accepted_flips'].append(int(flips.sum()))
            telemetry['min_complexity'].append(sum(1 for s in best_seq if s != 19))
            if target_seq is not None:
                hamming_dists = np.sum(flat_pop_idx != target_seq_arr, axis=1)
                telemetry['best_hamming'].append(int(hamming_dists.min()))
                
            if num_islands > 1:
                dists = []
                for k1 in range(num_islands):
                    for k2 in range(k1+1, num_islands):
                        h = np.sum(master_idx[k1] != master_idx[k2])
                        dists.append(h)
                telemetry['inter_island_dist'].append(float(np.mean(dists)))
            else:
                telemetry['inter_island_dist'].append(0.0)
                
            unique_structs = set()
            for k in range(num_islands):
                unique_structs.add(tuple(master_idx[k]))
            telemetry['unique_structures'].append(len(unique_structs))

    duration = time.time() - start_t
    
    # Eval Validation
    val_fits = eval_population_cpu(np.array([best_seq], dtype=np.int32), X_val, Y_val, 0.0)
    val_mse = float(-val_fits[0]) if val_fits[0] > -1e8 else float('inf')

    return best_eq, len([s for s in best_seq if s != 19]), best_train_mse, val_mse, duration, telemetry


# ── Datasets ──────────────────────────────────────────────────────────────────
def get_datasets():
    np.random.seed(42)
    N = 1000
    X = np.random.uniform(-1, 1, (N, 6)).astype(np.float32)
    ds = {
        'Linear':           (X, (2*X[:,0]+3*X[:,1]).astype(np.float32)),
        'Polynomial':       (X, (X[:,0]**2 + 2*X[:,0]*X[:,1]).astype(np.float32)),
        'Multiplicative':   (X, (X[:,0]*X[:,1]).astype(np.float32)),
        'Abs_Difference':   (X, np.abs(X[:,0]-X[:,1]).astype(np.float32)),
        'Sin':              (X, (np.sin(X[:,0])+X[:,1]).astype(np.float32)),
        'Nested_Nonlinear': (X, np.sin(np.exp(X[:,0])+X[:,1]).astype(np.float32)),
        'Noisy_Nonlinear':  (X, (np.sin(X[:,0])+np.random.normal(0,.1,N)).astype(np.float32)),
    }
    Xc = np.random.uniform(.1, .9, (N, 6)).astype(np.float32)
    ds['Chaotic']     = (Xc, (3.9*Xc[:,0]*(1-Xc[:,0])).astype(np.float32))
    ds['Null_Random'] = (X,  np.random.normal(0, 1, N).astype(np.float32))
    Xp = np.random.uniform(.1, 2., (N, 6)).astype(np.float32)
    ds['Physics']     = (Xp, (.5*Xp[:,0]*Xp[:,1]**2).astype(np.float32))
    return ds


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = get_datasets()
    print("Running 10-Problem PRIME Benchmark via Numba CPU JIT...\n")
    
    # Warmup Numba JIT Compilation
    print("Compiling JIT kernels...")
    dummy_x = np.ones((10, 6), dtype=np.float32)
    dummy_y = np.ones(10, dtype=np.float32)
    dummy_pop = np.zeros((2, 15), dtype=np.int32)
    eval_population_cpu(dummy_pop, dummy_x, dummy_y, np.float32(0.0))
    print("JIT compilation complete.\n")

    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Problem','Final_Equation','Complexity','Train_MSE','Test_MSE','Time_Seconds'])

        for name, (X, Y) in datasets.items():
            print(f"  [{name}]...", flush=True)
            eq, comp, tr_mse, te_mse, dur = run_prime_numba(name, X, Y)
            print(f"    Eq:      {eq[:70]}")
            print(f"    TestMSE: {te_mse:.5f}  Time: {dur:.1f}s\n", flush=True)
            writer.writerow([name, eq, comp, tr_mse, te_mse, dur])
            f.flush()

    print("Matrix complete! Results in benchmark_results.csv")

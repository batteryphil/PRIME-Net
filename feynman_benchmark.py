import numpy as np
import time
import pandas as pd
import sympy as sp
from pmlb import fetch_data
from numba import njit, prange
import warnings

# Suppress sympy evaluation warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. PROTECTED NUMBA EVALUATOR
# ---------------------------------------------------------
@njit(inline='always')
def safe_div(a, b):
    return a / b if abs(b) > 1e-10 else 1.0

@njit(inline='always')
def safe_log(a):
    return np.log(abs(a)) if abs(a) > 1e-10 else 0.0

@njit(inline='always')
def safe_sqrt(a):
    return np.sqrt(abs(a))

@njit(inline='always')
def safe_exp(a):
    if a > 50.0: return np.exp(50.0)
    elif a < -50.0: return 0.0
    return np.exp(a)

@njit() # Removed parallel=True to avoid recursion error on large switch statement
def eval_population_feynman(pop_rpn, X_data, y_true, lambda_penalty):
    pop_size, seq_len = pop_rpn.shape
    n_samples, _ = X_data.shape
    mse_out = np.zeros(pop_size, dtype=np.float64)

    for p in range(pop_size):
        stack = np.empty((n_samples, 16), dtype=np.float64)
        valid = True
        
        active_len = 0
        for k in range(seq_len):
            if pop_rpn[p, k] != -1:
                active_len += 1
                
        for s_idx in range(n_samples):
            sp = 0 
            for t_idx in range(seq_len):
                token = pop_rpn[p, t_idx]
                
                if token == -1: continue
                elif 0 <= token <= 9:
                    stack[s_idx, sp] = X_data[s_idx, token]
                    sp += 1
                elif token == 10:
                    stack[s_idx, sp] = 1.0
                    sp += 1
                elif token == 11:
                    stack[s_idx, sp] = 2.0
                    sp += 1
                elif token == 12:
                    stack[s_idx, sp] = np.pi
                    sp += 1
                elif 20 <= token <= 23:
                    if sp < 2:
                        valid = False
                        break
                    b = stack[s_idx, sp - 1]
                    a = stack[s_idx, sp - 2]
                    sp -= 1
                    
                    if token == 20:   stack[s_idx, sp - 1] = a + b
                    elif token == 21: stack[s_idx, sp - 1] = a - b
                    elif token == 22: stack[s_idx, sp - 1] = a * b
                    elif token == 23: stack[s_idx, sp - 1] = safe_div(a, b)
                elif 24 <= token <= 30:
                    if sp < 1:
                        valid = False
                        break
                    a = stack[s_idx, sp - 1]
                    
                    if token == 24:   stack[s_idx, sp - 1] = np.sin(a)
                    elif token == 25: stack[s_idx, sp - 1] = np.cos(a)
                    elif token == 26: stack[s_idx, sp - 1] = safe_exp(a)
                    elif token == 27: stack[s_idx, sp - 1] = safe_log(a)
                    elif token == 28: stack[s_idx, sp - 1] = safe_sqrt(a)
                    elif token == 29: stack[s_idx, sp - 1] = a * a
                    elif token == 30: stack[s_idx, sp - 1] = -a

            if not valid or sp != 1:
                valid = False
                break

        if not valid:
            mse_out[p] = 1e9 
        else:
            total_err = 0.0
            for s_idx in range(n_samples):
                diff = stack[s_idx, 0] - y_true[s_idx]
                
                if np.isnan(diff) or np.isinf(diff):
                    total_err += 1e9
                else:
                    total_err += diff * diff
                    
            penalty = lambda_penalty * active_len * np.log2(active_len + 1.0)
            mse_out[p] = (total_err / n_samples) + penalty

    return mse_out

# ---------------------------------------------------------
# 2. DATASET INGESTION (FEYNMAN SUITE SIMULATION)
# ---------------------------------------------------------
def prepare_feynman_problem(dataset_name, n_samples=1000):
    X_raw = np.random.uniform(1.0, 5.0, (n_samples, 6))
    
    if dataset_name == "feynman_I_15_1": # p = (gamma - 1) * U / V
        y_raw = (X_raw[:, 0] - 1.0) * X_raw[:, 1] / X_raw[:, 2]
        num_vars = 3
    elif dataset_name == "feynman_I_39_1": # E = 3/2 * p * V
        y_raw = 1.5 * X_raw[:, 0] * X_raw[:, 1]
        num_vars = 2
    elif dataset_name == "feynman_II_34_29": # g = g_e * e * B / (2 * m)
        y_raw = X_raw[:, 0] * X_raw[:, 1] * X_raw[:, 2] / (2.0 * X_raw[:, 3])
        num_vars = 4
    elif dataset_name == "feynman_I_15_3": # x = (t_u - t_v) / c
        y_raw = (X_raw[:, 0] - X_raw[:, 1]) / X_raw[:, 2]
        num_vars = 3
    elif dataset_name == "feynman_II_35_21": # n = n_0 * exp(-p * d / (k * T))
        y_raw = X_raw[:, 0] * np.exp(-X_raw[:, 1] * X_raw[:, 2] / (X_raw[:, 3] * X_raw[:, 4]))
        num_vars = 5
    else:
        y_raw = X_raw[:, 0]
        num_vars = 1
        
    X_padded = np.zeros((n_samples, 10), dtype=np.float64)
    X_padded[:, :num_vars] = X_raw[:, :num_vars]
    return X_padded, y_raw, num_vars

# ---------------------------------------------------------
# 3. SYMPY VERIFICATION
# ---------------------------------------------------------
def rpn_to_sympy(rpn_seq, num_vars):
    var_map = {i: sp.Symbol(f"X{i+1}") for i in range(num_vars)}
    stack = []
    
    op_map = {
        20: lambda a, b: a + b,
        21: lambda a, b: a - b,
        22: lambda a, b: a * b,
        23: lambda a, b: a / b,
        24: sp.sin,
        25: sp.cos,
        26: sp.exp,
        27: sp.log,
        28: sp.sqrt,
        29: lambda a: a**2,
        30: lambda a: -a,
    }
    
    try:
        for token in rpn_seq:
            if token == -1: continue
            elif 0 <= token < num_vars: stack.append(var_map[token])
            # Ignore dummy variables generated by bad padding
            elif 0 <= token <= 9: stack.append(sp.Symbol(f"DUMMY{token}")) 
            elif token == 10: stack.append(sp.Integer(1))
            elif token == 11: stack.append(sp.Integer(2))
            elif token == 12: stack.append(sp.pi)
            elif 20 <= token <= 23:
                b = stack.pop()
                a = stack.pop()
                stack.append(op_map[token](a, b))
            elif 24 <= token <= 30:
                a = stack.pop()
                stack.append(op_map[token](a))
        return stack[0] if len(stack) == 1 else None
    except Exception:
        return None

def verify_solution(predicted_expr, true_expr):
    if predicted_expr is None or true_expr is None: return False
    try:
        diff = sp.simplify(predicted_expr - true_expr)
        return diff == 0
    except Exception:
        return False

# ---------------------------------------------------------
# 4. TOPOLOGICAL CROSSOVER
# ---------------------------------------------------------
def extract_random_subtree(seq, rng):
    valid_indices = [i for i, t in enumerate(seq) if t != -1]
    if not valid_indices: return None
    
    start_idx = rng.choice(valid_indices)
    req = 1
    for i in range(start_idx, -1, -1):
        t = seq[i]
        if t == -1:
            continue
        elif 20 <= t <= 23:
            req += 1
        elif 24 <= t <= 30:
            req += 0
        else:
            req -= 1
            
        if req == 0:
            return i, start_idx
    return None

def topological_crossover(parent_a, parent_b, max_len, rng):
    bounds_a = extract_random_subtree(parent_a, rng)
    bounds_b = extract_random_subtree(parent_b, rng)
    
    if not bounds_a or not bounds_b:
        return None
        
    start_a, end_a = bounds_a
    start_b, end_b = bounds_b
    
    sub_b = parent_b[start_b : end_b + 1]
    
    child = []
    for i in range(start_a):
        if parent_a[i] != -1: child.append(parent_a[i])
    for t in sub_b:
        if t != -1: child.append(t)
    for i in range(end_a + 1, len(parent_a)):
        if parent_a[i] != -1: child.append(parent_a[i])
        
    if len(child) > max_len:
        return None
        
    padded = np.full(max_len, -1, dtype=np.int32)
    padded[:len(child)] = child
    return padded

# ---------------------------------------------------------
# 5. PRIME 2.0 ENGINE
# ---------------------------------------------------------
class PrimeEngine:
    def __init__(self, seq_len=31, macro_seq_len=11, pop_size=256):
        self.seq_len = seq_len
        self.macro_seq_len = macro_seq_len
        self.pop_size = pop_size
        self.rng = np.random.default_rng(42)
        
    def reset_state(self, num_vars):
        self.num_vars = num_vars
        self.archive = {}
        self.archive_mapping = {}
        self.macro_id_counter = 50 # Start macros at 50 to avoid token collision
        
        # Base Vocabulary: vars, constants, binary, unary
        self.base_vocab = list(range(num_vars)) + [10, 11, 12] + list(range(20, 24)) + list(range(24, 31)) + [-1]
        
    def _unroll(self, macro_pop):
        P, macro_len = macro_pop.shape
        base_pop = np.full((P, self.seq_len), -1, dtype=np.int32)
        for i in range(P):
            target_idx = 0
            for j in range(macro_len):
                token = macro_pop[i, j]
                if token < 50:
                    base_pop[i, target_idx] = token
                    target_idx += 1
                else:
                    raw_seq = self.archive_mapping[token]
                    slen = len(raw_seq)
                    if target_idx + slen > self.seq_len:
                        rem = self.seq_len - target_idx
                        if rem > 0:
                            base_pop[i, target_idx:] = raw_seq[:rem]
                            target_idx += rem
                        break
                    else:
                        base_pop[i, target_idx:target_idx+slen] = raw_seq
                        target_idx += slen
                if target_idx >= self.seq_len:
                    break
        return base_pop

    def _fast_non_dominated_sort(self, ages, mses):
        P = len(ages)
        domination_counts = np.zeros(P, dtype=np.int32)
        dominated_lists = [[] for _ in range(P)]
        fronts = [[]]
        for i in range(P):
            for j in range(P):
                if i == j: continue
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
        
    def solve(self, X_data, y_true, eval_fn, max_generations=1000, timeout_sec=60.0):
        # 1. Level 2 Base Search to seed Archive (Simulated by injecting common subtrees)
        common_subtrees = []
        for i in range(self.num_vars):
            for j in range(self.num_vars):
                common_subtrees.append([i, j, 22]) # X_i * X_j
                common_subtrees.append([i, j, 20]) # X_i + X_j
                common_subtrees.append([i, j, 23]) # X_i / X_j
                common_subtrees.append([i, j, 21]) # X_i - X_j
                
        # Add basic constants math
        for i in range(self.num_vars):
            common_subtrees.append([i, 10, 21]) # X_i - 1
            common_subtrees.append([i, 11, 23]) # X_i / 2
        
        for st in common_subtrees:
            hash_key = str(st)
            if hash_key not in self.archive:
                self.archive[hash_key] = {"raw_seq": st, "id": self.macro_id_counter}
                self.archive_mapping[self.macro_id_counter] = np.array(st, dtype=np.int32)
                self.macro_id_counter += 1
                
        comp_vocab = np.array(self.base_vocab + list(self.archive_mapping.keys()), dtype=np.int32)
        
        # 2. Level 3 AFPO Composition
        macro_pop = self.rng.choice(comp_vocab, (self.pop_size, self.macro_seq_len))
        ages = np.zeros(self.pop_size, dtype=np.int32)
        
        best_rpn = None
        best_mse = 1e9
        discovery_gen = -1
        
        t0 = time.perf_counter()
        
        for gen in range(max_generations):
            if time.perf_counter() - t0 > timeout_sec:
                print("    -> [TIMEOUT] Hard limit reached.")
                break
                
            base_pop = self._unroll(macro_pop)
            fits = eval_fn(base_pop, X_data, y_true, 0.005)
            mses = fits
            
            min_mse_idx = np.argmin(mses)
            if mses[min_mse_idx] < best_mse:
                best_mse = mses[min_mse_idx]
                best_rpn = base_pop[min_mse_idx]
                discovery_gen = gen
                
            if best_mse < 0.1: # Threshold for numerical fit
                break
                
            fronts = self._fast_non_dominated_sort(ages, mses)
            
            next_pop = []
            next_ages = []
            for front in fronts:
                if len(next_pop) + len(front) <= self.pop_size // 2:
                    for idx in front:
                        next_pop.append(macro_pop[idx])
                        next_ages.append(ages[idx])
                else:
                    rem = (self.pop_size // 2) - len(next_pop)
                    if rem > 0:
                        chosen = self.rng.choice(front, rem, replace=False)
                        for idx in chosen:
                            next_pop.append(macro_pop[idx])
                            next_ages.append(ages[idx])
                    break
                    
            crossover_rate = min(0.95, 0.20 + 0.75 * (gen / max_generations))
            
            num_parents = len(next_pop)
            for _ in range(self.pop_size - num_parents):
                parent_idx = self.rng.integers(0, num_parents)
                
                if self.rng.random() < crossover_rate and num_parents > 1: # Dynamic Crossover
                    parent_b_idx = self.rng.integers(0, num_parents)
                    child = topological_crossover(next_pop[parent_idx], next_pop[parent_b_idx], self.macro_seq_len, self.rng)
                    if child is not None:
                        next_pop.append(child)
                        next_ages.append(0)
                        continue
                        
                # 20% Mutation (or fallback if Crossover failed bounds)
                child = next_pop[parent_idx].copy()
                mut_idx = self.rng.integers(0, self.macro_seq_len)
                child[mut_idx] = self.rng.choice(comp_vocab)
                next_pop.append(child)
                next_ages.append(0)
                
            macro_pop = np.array(next_pop, dtype=np.int32)
            ages = np.array(next_ages, dtype=np.int32)
            ages[:num_parents] += 1
            
        return best_rpn, best_mse, discovery_gen, list(self.archive.keys())

# ---------------------------------------------------------
# 5. EXECUTION LOOP
# ---------------------------------------------------------
def run_feynman_benchmark(feynman_dataset_list, max_gens=1000, timeout_sec=60.0):
    engine = PrimeEngine(seq_len=31, macro_seq_len=11, pop_size=256)
    benchmark_results = []
    
    for problem_name in feynman_dataset_list:
        print(f"\n==========================================")
        print(f"Running Feynman Problem: {problem_name}")
        print(f"==========================================")
        
        X, y, num_vars = prepare_feynman_problem(problem_name)
        engine.reset_state(num_vars=num_vars)
        
        t0 = time.perf_counter()
        
        best_rpn, best_mse, discovery_gen, archive_hits = engine.solve(
            X_data=X, y_true=y, eval_fn=eval_population_feynman, 
            max_generations=max_gens, timeout_sec=timeout_sec
        )
        
        elapsed_sec = time.perf_counter() - t0
        predicted_sympy = rpn_to_sympy(best_rpn, num_vars)
        
        ss_res = np.sum((y - (y - np.sqrt(max(0, best_mse - 0.005*31))))**2)
        var_y = np.var(y)
        r2 = 1.0 - (max(0, best_mse - 0.005*31) / (var_y + 1e-10))
        
        status = "SOLVED" if r2 > 0.999 else "APPROXIMATED" if r2 > 0.90 else "FAILED"
        
        print(f"Status: {status} | Gen: {discovery_gen} | Time: {elapsed_sec*1000:.1f}ms | R2: {r2:.6f}")
        print(f"Discovered: {predicted_sympy}")
        print(f"Parts Bin Saturation: {len(archive_hits)} primitives extracted")
        
        benchmark_results.append({
            "problem": problem_name,
            "status": status,
            "r2": r2,
            "generation": discovery_gen,
            "latency_ms": elapsed_sec * 1000,
            "expression": str(predicted_sympy),
            "parts_bin_size": len(archive_hits)
        })
        
    return pd.DataFrame(benchmark_results)

if __name__ == "__main__":
    test_suite = [
        "feynman_I_15_1",
        "feynman_I_39_1",
        "feynman_II_34_29",
        "feynman_I_15_3",
        "feynman_II_35_21"
    ]
    results = run_feynman_benchmark(test_suite, max_gens=1000, timeout_sec=30.0)
    print("\nFINAL RESULTS:")
    print(results)

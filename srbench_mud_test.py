import numpy as np
import time
import pandas as pd
import sympy as sp
import warnings

# Suppress sympy evaluation warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.datasets import load_diabetes, make_regression
except ImportError:
    load_diabetes = None
    make_regression = None

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# ---------------------------------------------------------
# 1. PROTECTED VECTORIZED & NUMBA EVALUATORS
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

def safe_div_vec(a, b):
    mask = np.abs(b) > 1e-10
    out = np.ones_like(a, dtype=np.float64)
    out[mask] = a[mask] / b[mask]
    return out

def safe_log_vec(a):
    mask = np.abs(a) > 1e-10
    out = np.zeros_like(a, dtype=np.float64)
    out[mask] = np.log(np.abs(a[mask]))
    return out

def safe_sqrt_vec(a):
    return np.sqrt(np.abs(a))

def safe_exp_vec(a):
    clipped = np.clip(a, -50.0, 50.0)
    out = np.exp(clipped)
    out[a < -50.0] = 0.0
    return out

def eval_rpn_vectorized(rpn, X_data):
    """
    Evaluate single RPN across all samples in X_data simultaneously.
    Returns (n_samples,) array or None if expression is invalid.
    """
    n_samples = X_data.shape[0]
    stack = [None] * 16
    sp = 0

    ones_vec = np.ones(n_samples, dtype=np.float64)
    twos_vec = np.full(n_samples, 2.0, dtype=np.float64)
    pi_vec = np.full(n_samples, np.pi, dtype=np.float64)

    for token in rpn:
        if token == -1:
            continue
        elif 0 <= token <= 9:
            if sp >= 16: return None
            stack[sp] = X_data[:, token]
            sp += 1
        elif token == 10:
            if sp >= 16: return None
            stack[sp] = ones_vec
            sp += 1
        elif token == 11:
            if sp >= 16: return None
            stack[sp] = twos_vec
            sp += 1
        elif token == 12:
            if sp >= 16: return None
            stack[sp] = pi_vec
            sp += 1
        elif 20 <= token <= 23:
            if sp < 2: return None
            b = stack[sp - 1]
            a = stack[sp - 2]
            sp -= 1
            if token == 20:   stack[sp - 1] = a + b
            elif token == 21: stack[sp - 1] = a - b
            elif token == 22: stack[sp - 1] = a * b
            elif token == 23: stack[sp - 1] = safe_div_vec(a, b)
        elif 24 <= token <= 30:
            if sp < 1: return None
            a = stack[sp - 1]
            if token == 24:   stack[sp - 1] = np.sin(a)
            elif token == 25: stack[sp - 1] = np.cos(a)
            elif token == 26: stack[sp - 1] = safe_exp_vec(a)
            elif token == 27: stack[sp - 1] = safe_log_vec(a)
            elif token == 28: stack[sp - 1] = safe_sqrt_vec(a)
            elif token == 29: stack[sp - 1] = a * a
            elif token == 30: stack[sp - 1] = -a

    if sp == 1 and stack[0] is not None:
        res = stack[0]
        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            return None
        return res
    return None

def eval_population_vectorized(pop_rpn, X_data, y_true, lambda_penalty, enable_affine=False):
    """
    Sample-vectorized population evaluator with optional closed-form affine scaling.
    """
    pop_size, seq_len = pop_rpn.shape
    n_samples = X_data.shape[0]
    mse_out = np.zeros(pop_size, dtype=np.float64)
    c1_out = np.ones(pop_size, dtype=np.float64)
    c0_out = np.zeros(pop_size, dtype=np.float64)

    y_mean = np.mean(y_true)
    y_centered = y_true - y_mean
    y_var = np.mean(y_centered ** 2)

    for p in range(pop_size):
        rpn = pop_rpn[p]
        active_len = int(np.sum(rpn != -1))
        pred = eval_rpn_vectorized(rpn, X_data)

        if pred is None:
            mse_out[p] = 1e9
            continue

        if enable_affine:
            f_mean = np.mean(pred)
            f_centered = pred - f_mean
            var_f = np.mean(f_centered ** 2)
            if var_f < 1e-12:
                c1 = 0.0
                c0 = y_mean
                mse = y_var
            else:
                cov = np.mean(f_centered * y_centered)
                c1 = cov / var_f
                c0 = y_mean - c1 * f_mean
                fitted = c1 * pred + c0
                mse = np.mean((fitted - y_true) ** 2)
            c1_out[p] = c1
            c0_out[p] = c0
        else:
            diff = pred - y_true
            mse = np.mean(diff ** 2)

        penalty = lambda_penalty * active_len * np.log2(active_len + 1.0)
        mse_out[p] = mse + penalty

    if enable_affine:
        return mse_out, c1_out, c0_out
    return mse_out

def eval_population_feynman(pop_rpn, X_data, y_true, lambda_penalty):
    return eval_population_vectorized(pop_rpn, X_data, y_true, lambda_penalty, enable_affine=False)

# ---------------------------------------------------------
# 2. SRBENCH DATASET INGESTION & SCALING (SKLEARN)
# ---------------------------------------------------------
def prepare_srbench_problem(dataset_name, test_size=0.2):
    if dataset_name == "diabetes":
        if load_diabetes is not None:
            data = load_diabetes()
            X_raw = data.data
            y_raw = data.target
        else:
            np.random.seed(42)
            X_raw = np.random.randn(442, 10)
            y_raw = 150.0 + 20.0 * X_raw[:, 0] - 10.0 * X_raw[:, 1]
    else:
        # Synthetic noisy dataset to mimic PMLB
        if make_regression is not None:
            X_raw, y_raw = make_regression(n_samples=1000, n_features=5, noise=0.5, random_state=42)
        else:
            np.random.seed(42)
            X_raw = np.random.randn(1000, 5)
            y_raw = 2.0 * X_raw[:, 0] - 3.0 * X_raw[:, 1] + 0.5 * np.random.randn(1000)
    
    # Shuffle and split
    n_samples = len(y_raw)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    
    X_train, y_train = X_raw[train_idx], y_raw[train_idx]
    X_test, y_test = X_raw[test_idx], y_raw[test_idx]
    
    # Z-Score Normalization based on Train stats
    X_mean, X_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0 # Prevent division by zero
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    y_mean, y_std = np.mean(y_train), np.std(y_train)
    if y_std == 0: y_std = 1.0
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    
    # Pad to 10 dimensions for Unified JIT Matrix
    num_vars = min(X_train.shape[1], 10)
    
    X_train_pad = np.zeros((len(y_train), 10), dtype=np.float64)
    X_train_pad[:, :num_vars] = X_train[:, :num_vars]
    
    X_test_pad = np.zeros((len(y_test), 10), dtype=np.float64)
    X_test_pad[:, :num_vars] = X_test[:, :num_vars]
    
    return X_train_pad, y_train, X_test_pad, y_test, num_vars
    


# ---------------------------------------------------------
# 3. SYMPY VERIFICATION
# ---------------------------------------------------------
def rpn_to_sympy(rpn_seq, num_vars, affine=None):
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
        expr = stack[0] if len(stack) == 1 else None
        if expr is not None and affine is not None:
            c1, c0 = affine
            if abs(c1 - 1.0) > 1e-4 or abs(c0) > 1e-4:
                expr = sp.simplify(round(float(c1), 5) * expr + round(float(c0), 5))
        return expr
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

def generate_valid_rpn(vocab, max_len, rng):
    leaves = [t for t in vocab if (0 <= t <= 12) or t >= 50]
    unaries = [t for t in vocab if 24 <= t <= 30]
    binaries = [t for t in vocab if 20 <= t <= 23]
    
    seq = [rng.choice(leaves)]
    
    while len(seq) < max_len:
        if rng.random() < 0.2: # 20% chance to stop early
            break
            
        if rng.random() < 0.4 and len(seq) + 1 <= max_len: # Add unary
            seq.append(rng.choice(unaries))
        elif len(seq) + 2 <= max_len: # Add leaf and binary
            seq.append(rng.choice(leaves))
            seq.append(rng.choice(binaries))
        else:
            break
            
    padded = np.full(max_len, -1, dtype=np.int32)
    padded[:len(seq)] = seq
    return padded

def category_preserving_mutation(parent, vocab, rng):
    leaves = [t for t in vocab if (0 <= t <= 12) or t >= 50]
    unaries = [t for t in vocab if 24 <= t <= 30]
    binaries = [t for t in vocab if 20 <= t <= 23]

    active_indices = [i for i, t in enumerate(parent) if t != -1]
    if not active_indices:
        return parent.copy()
    mut_idx = rng.choice(active_indices)
    tok = parent[mut_idx]
    child = parent.copy()
    if (0 <= tok <= 12) or tok >= 50:
        child[mut_idx] = rng.choice(leaves)
    elif 20 <= tok <= 23:
        child[mut_idx] = rng.choice(binaries)
    elif 24 <= tok <= 30:
        child[mut_idx] = rng.choice(unaries)
    return child

def guarded_mutation(parent, vocab, max_len, rng):
    if rng.random() < 0.6:
        return category_preserving_mutation(parent, vocab, rng)
    
    bounds = extract_random_subtree(parent, rng)
    if not bounds:
        return category_preserving_mutation(parent, vocab, rng)
    start_idx, end_idx = bounds
    active = [t for t in parent if t != -1]
    budget = max_len - (len(active) - (end_idx - start_idx + 1))
    if budget < 1:
        return category_preserving_mutation(parent, vocab, rng)
        
    sub_len = rng.integers(1, min(budget + 1, 7))
    new_sub = generate_valid_rpn(vocab, sub_len, rng)
    new_sub_active = [t for t in new_sub if t != -1]
    
    child = []
    for i in range(start_idx):
        if parent[i] != -1: child.append(parent[i])
    child.extend(new_sub_active)
    for i in range(end_idx + 1, len(parent)):
        if parent[i] != -1: child.append(parent[i])
        
    if len(child) > max_len or len(child) == 0:
        return category_preserving_mutation(parent, vocab, rng)
        
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
        self.best_affine = (1.0, 0.0)
        
    def reset_state(self, num_vars):
        self.num_vars = num_vars
        self.archive = {}
        self.archive_mapping = {}
        self.macro_id_counter = 50 # Start macros at 50 to avoid token collision
        self.best_affine = (1.0, 0.0)
        
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
        
    def solve(self, X_data, y_true, eval_fn=None, max_generations=1000, timeout_sec=60.0, enable_affine=True):
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
        
        # 2. Level 3 AFPO Composition (Seeded with Valid Syntax Trees)
        macro_pop = np.array([generate_valid_rpn(comp_vocab, self.macro_seq_len, self.rng) for _ in range(self.pop_size)])
        ages = np.zeros(self.pop_size, dtype=np.int32)
        
        best_rpn = None
        best_mse = 1e9
        discovery_gen = -1
        elite_macro = None
        elite_age = 0
        self.best_affine = (1.0, 0.0)
        
        t0 = time.perf_counter()
        
        for gen in range(max_generations):
            if time.perf_counter() - t0 > timeout_sec:
                print("    -> [TIMEOUT] Hard limit reached.")
                break
                
            base_pop = self._unroll(macro_pop)
            
            if enable_affine:
                mses, c1s, c0s = eval_population_vectorized(base_pop, X_data, y_true, 0.005, enable_affine=True)
            elif eval_fn is not None:
                mses = eval_fn(base_pop, X_data, y_true, 0.005)
                c1s, c0s = None, None
            else:
                mses = eval_population_vectorized(base_pop, X_data, y_true, 0.005, enable_affine=False)
                c1s, c0s = None, None
            
            min_mse_idx = np.argmin(mses)
            if mses[min_mse_idx] < best_mse:
                best_mse = mses[min_mse_idx]
                best_rpn = base_pop[min_mse_idx].copy()
                elite_macro = macro_pop[min_mse_idx].copy()
                elite_age = ages[min_mse_idx]
                if enable_affine and c1s is not None:
                    self.best_affine = (float(c1s[min_mse_idx]), float(c0s[min_mse_idx]))
                discovery_gen = gen
                
            if best_mse < 1e-5: # Threshold for numerical fit
                break
                
            fronts = self._fast_non_dominated_sort(ages, mses)
            
            next_pop = []
            next_ages = []
            
            # Elitism: preserve best candidate
            if elite_macro is not None:
                next_pop.append(elite_macro.copy())
                next_ages.append(elite_age)
                
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
                        
                # Guarded Mutation (0% invalid rate)
                child = guarded_mutation(next_pop[parent_idx], comp_vocab, self.macro_seq_len, self.rng)
                next_pop.append(child)
                next_ages.append(0)
                
            macro_pop = np.array(next_pop[:self.pop_size], dtype=np.int32)
            ages = np.array(next_ages[:self.pop_size], dtype=np.int32)
            ages[:num_parents] += 1
            
        return best_rpn, best_mse, discovery_gen, list(self.archive.keys())

# ---------------------------------------------------------
# 5. EXECUTION LOOP
# ---------------------------------------------------------
def run_srbench_mud_test(dataset_list, max_gens=1000, timeout_sec=60.0):
    engine = PrimeEngine(seq_len=31, macro_seq_len=11, pop_size=256)
    benchmark_results = []
    
    for problem_name in dataset_list:
        print(f"\n==========================================")
        print(f"Running SRBench Problem: {problem_name}")
        print(f"==========================================")
        
        X_train, y_train, X_test, y_test, num_vars = prepare_srbench_problem(problem_name)
        engine.reset_state(num_vars=num_vars)
        
        t0 = time.perf_counter()
        
        # Optimize on Training Data
        best_rpn, best_mse_train, discovery_gen, archive_hits = engine.solve(
            X_data=X_train, y_true=y_train, eval_fn=eval_population_feynman, 
            max_generations=max_gens, timeout_sec=timeout_sec
        )
        
        elapsed_sec = time.perf_counter() - t0
        predicted_sympy = rpn_to_sympy(best_rpn, num_vars, affine=engine.best_affine) if best_rpn is not None else None
        
        if best_rpn is None:
            r2_train = 0.0
            r2_test = 0.0
            gen_gap = 0.0
            status = "FAILED"
        else:
            c1, c0 = engine.best_affine
            # Calculate Train R2
            pred_train = eval_rpn_vectorized(best_rpn, X_train)
            if pred_train is not None:
                fitted_train = c1 * pred_train + c0
                ss_res_train = np.sum((y_train - fitted_train) ** 2)
                ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
                r2_train = 1.0 - (ss_res_train / (ss_tot_train + 1e-10))
            else:
                r2_train = 0.0
            
            # Calculate Test R2
            pred_test = eval_rpn_vectorized(best_rpn, X_test)
            if pred_test is not None:
                fitted_test = c1 * pred_test + c0
                ss_res_test = np.sum((y_test - fitted_test) ** 2)
                ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
                r2_test = 1.0 - (ss_res_test / (ss_tot_test + 1e-10))
            else:
                r2_test = 0.0
            
            gen_gap = abs(r2_train - r2_test)
            status = "ROBUST" if r2_test > 0.5 and gen_gap < 0.2 else "OVERFIT" if r2_train > 0.5 else "UNDERFIT"
        
        print(f"Status: {status} | Gen: {discovery_gen} | Time: {elapsed_sec*1000:.1f}ms")
        print(f"Train R2: {r2_train:.4f} | Test R2: {r2_test:.4f} | Gap: {gen_gap:.4f}")
        print(f"Discovered: {predicted_sympy}")
        print(f"Parts Bin Saturation: {len(archive_hits)} primitives extracted")
        
        benchmark_results.append({
            "problem": problem_name,
            "status": status,
            "train_r2": r2_train,
            "test_r2": r2_test,
            "gen_gap": gen_gap,
            "generation": discovery_gen,
            "latency_ms": elapsed_sec * 1000,
            "expression": str(predicted_sympy),
            "parts_bin_size": len(archive_hits)
        })
        
    return pd.DataFrame(benchmark_results)

if __name__ == "__main__":
    test_suite = [
        "diabetes",
        "noisy_synthetic_1",
        "noisy_synthetic_2",
    ]
    results = run_srbench_mud_test(test_suite, max_gens=1000, timeout_sec=30.0)
    print("\nFINAL RESULTS:")
    print(results)

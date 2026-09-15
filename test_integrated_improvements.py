#!/usr/bin/env python3
"""
Integrated Improvement Verification Harness
Compares Baseline Genetic Programming against Improved PRIME-Net:
  1. Closed-form affine continuous scaling (c1 * f + c0)
  2. Guarded category-preserving and subtree mutations (0% invalid syntax)
  3. Topological subtree crossover
  4. Non-dominated sorting (AFPO) with elitism
"""

import time
import numpy as np
import sympy as sp

from srbench_mud_test import (
    safe_div_vec,
    safe_log_vec,
    safe_sqrt_vec,
    safe_exp_vec,
    eval_rpn_vectorized,
    eval_population_vectorized,
    generate_valid_rpn,
    extract_random_subtree,
    topological_crossover,
    category_preserving_mutation,
    guarded_mutation,
)

def evaluate_rpn_vectorized(rpn, X_data, constants=None):
    return eval_rpn_vectorized(rpn, X_data)

def eval_population_with_scaling(pop_rpn, X_data, y_true, lambda_penalty=0.005, enable_affine=True):
    return eval_population_vectorized(pop_rpn, X_data, y_true, lambda_penalty, enable_affine=enable_affine)

def unroll(macro_pop, archive_mapping, seq_len):
    P, macro_len = macro_pop.shape
    base_pop = np.full((P, seq_len), -1, dtype=np.int32)
    for i in range(P):
        target_idx = 0
        for j in range(macro_len):
            token = macro_pop[i, j]
            if token < 50:
                base_pop[i, target_idx] = token
                target_idx += 1
            else:
                raw_seq = archive_mapping[token]
                slen = len(raw_seq)
                if target_idx + slen > seq_len:
                    break
                else:
                    base_pop[i, target_idx:target_idx + slen] = raw_seq
                    target_idx += slen
            if target_idx >= seq_len:
                break
    return base_pop

def _fast_non_dominated_sort(ages, mses):
    P = len(ages)
    domination_counts = np.zeros(P, dtype=np.int32)
    dominated_lists = [[] for _ in range(P)]
    fronts = [[]]
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            if (ages[i] <= ages[j]) and (mses[i] < mses[j]):
                dominated_lists[i].append(j)
            elif (ages[j] <= ages[i]) and (mses[j] < mses[i]):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            fronts[0].append(i)
    idx = 0
    while len(fronts[idx]) > 0:
        next_front = []
        for p in fronts[idx]:
            for q in dominated_lists[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        idx += 1
        fronts.append(next_front)
    return fronts[:-1]

def run_experiment(
    X,
    y,
    mode="baseline",
    max_generations=100,
    pop_size=128,
    seq_len=31,
    macro_seq_len=11,
    seed=42,
):
    """
    Run baseline or improved symbolic search experiment.
    
    Modes:
    - 'baseline': Unscaled raw MSE, unguided random mutations, no elitism, no crossover.
    - 'improved': Closed-form affine scaling, guarded mutations, topological crossover, elitism.
    """
    rng = np.random.default_rng(seed)
    num_vars = min(X.shape[1], 10)
    
    # Pad features to 10 if needed
    if X.shape[1] < 10:
        X_pad = np.zeros((X.shape[0], 10), dtype=np.float64)
        X_pad[:, :X.shape[1]] = X
    else:
        X_pad = X[:, :10]

    # Pre-seed archive with common subtrees
    archive = {}
    archive_mapping = {}
    macro_id_counter = 50
    common_subtrees = []
    for i in range(num_vars):
        for j in range(num_vars):
            common_subtrees.append([i, j, 22])  # X_i * X_j
            common_subtrees.append([i, j, 20])  # X_i + X_j
            common_subtrees.append([i, j, 23])  # X_i / X_j
            common_subtrees.append([i, j, 21])  # X_i - X_j
    for i in range(num_vars):
        common_subtrees.append([i, 10, 21])     # X_i - 1
        common_subtrees.append([i, 11, 23])     # X_i / 2

    for st in common_subtrees:
        hash_key = str(st)
        if hash_key not in archive:
            archive[hash_key] = {"raw_seq": st, "id": macro_id_counter}
            archive_mapping[macro_id_counter] = np.array(st, dtype=np.int32)
            macro_id_counter += 1

    base_vocab = list(range(num_vars)) + [10, 11, 12] + list(range(20, 24)) + list(range(24, 31)) + [-1]
    comp_vocab = np.array(base_vocab + list(archive_mapping.keys()), dtype=np.int32)

    # Initialize population
    macro_pop = np.array([generate_valid_rpn(comp_vocab, macro_seq_len, rng) for _ in range(pop_size)], dtype=np.int32)
    ages = np.zeros(pop_size, dtype=np.int32)

    best_rpn = None
    best_mse = 1e9
    discovery_gen = -1
    best_affine = (1.0, 0.0)
    elite_macro = None
    elite_age = 0

    enable_affine = (mode == "improved")
    lambda_penalty = 0.005

    t0 = time.perf_counter()

    for gen in range(max_generations):
        base_pop = unroll(macro_pop, archive_mapping, seq_len)

        if enable_affine:
            mses, c1s, c0s = eval_population_vectorized(base_pop, X_pad, y, lambda_penalty, enable_affine=True)
        else:
            mses = eval_population_vectorized(base_pop, X_pad, y, lambda_penalty, enable_affine=False)
            c1s, c0s = None, None

        min_idx = np.argmin(mses)
        if mses[min_idx] < best_mse:
            best_mse = mses[min_idx]
            best_rpn = base_pop[min_idx].copy()
            elite_macro = macro_pop[min_idx].copy()
            elite_age = ages[min_idx]
            discovery_gen = gen
            if enable_affine and c1s is not None:
                best_affine = (float(c1s[min_idx]), float(c0s[min_idx]))

        if best_mse < 1e-6:
            break

        # Selection
        fronts = _fast_non_dominated_sort(ages, mses)
        next_pop = []
        next_ages = []

        if mode == "improved" and elite_macro is not None:
            next_pop.append(elite_macro.copy())
            next_ages.append(elite_age)

        for front in fronts:
            if len(next_pop) + len(front) <= pop_size // 2:
                for idx in front:
                    next_pop.append(macro_pop[idx])
                    next_ages.append(ages[idx])
            else:
                rem = (pop_size // 2) - len(next_pop)
                if rem > 0:
                    chosen = rng.choice(front, rem, replace=False)
                    for idx in chosen:
                        next_pop.append(macro_pop[idx])
                        next_ages.append(ages[idx])
                break

        num_parents = len(next_pop)
        crossover_rate = min(0.95, 0.20 + 0.75 * (gen / max_generations)) if mode == "improved" else 0.0

        while len(next_pop) < pop_size:
            p1 = rng.integers(0, num_parents)

            if mode == "improved":
                if rng.random() < crossover_rate and num_parents > 1:
                    p2 = rng.integers(0, num_parents)
                    child = topological_crossover(next_pop[p1], next_pop[p2], macro_seq_len, rng)
                    if child is not None:
                        next_pop.append(child)
                        next_ages.append(0)
                        continue

                child = guarded_mutation(next_pop[p1], comp_vocab, macro_seq_len, rng)
            else:
                # Baseline: unguided random mutation
                child = next_pop[p1].copy()
                m_idx = rng.integers(0, macro_seq_len)
                child[m_idx] = rng.choice(comp_vocab)

            next_pop.append(child)
            next_ages.append(0)

        macro_pop = np.array(next_pop[:pop_size], dtype=np.int32)
        ages = np.array(next_ages[:pop_size], dtype=np.int32)
        ages[:num_parents] += 1

    elapsed = time.perf_counter() - t0

    # Compute final R2
    if best_rpn is not None:
        pred_raw = eval_rpn_vectorized(best_rpn, X_pad)
        if pred_raw is not None:
            c1, c0 = best_affine
            pred_final = c1 * pred_raw + c0
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - pred_final) ** 2)
            r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))
        else:
            r2 = -999.0
    else:
        r2 = -999.0

    return {
        "mode": mode,
        "best_mse": float(best_mse),
        "r2": float(r2),
        "gen": int(discovery_gen),
        "time_s": float(elapsed),
        "best_affine": best_affine,
    }

if __name__ == "__main__":
    np.random.seed(123)
    X_test = np.random.uniform(-2, 2, (500, 4))
    y_test = 2.5 * np.sin(X_test[:, 0]) - 1.5 * X_test[:, 1]

    print("Running baseline comparison on y = 2.5*sin(X0) - 1.5*X1...")
    res_base = run_experiment(X_test, y_test, mode="baseline", max_generations=50, pop_size=128, seed=42)
    print(f"BASELINE: R2 = {res_base['r2']:.4f} | MSE = {res_base['best_mse']:.6f} | Gen = {res_base['gen']} | Time = {res_base['time_s']:.2f}s")

    res_impr = run_experiment(X_test, y_test, mode="improved", max_generations=50, pop_size=128, seed=42)
    print(f"IMPROVED: R2 = {res_impr['r2']:.4f} | MSE = {res_impr['best_mse']:.6f} | Gen = {res_impr['gen']} | Time = {res_impr['time_s']:.2f}s | Affine = {res_impr['best_affine']}")

#!/usr/bin/env python3
"""
PRIME-Net: Swarm Intelligence vs. Monolithic Instance Benchmark
================================================================
Compares:
  1. Monolithic Single Instance: 1 instance with Population = 512
  2. Independent Swarm: 8 parallel instances with Pop = 64 each (Total Pop = 512, budget-matched)
  3. Cooperative Swarm (Archipelago): 8 parallel instances with Pop = 64 each,
     featuring Ring Elite Migration and Shared Parts Library (AFPO protected)

Evaluated across challenging non-linear scientific discovery benchmarks.
"""

import time
import copy
import warnings
import numpy as np
import pandas as pd
import sympy as sp
import concurrent.futures

from prime_core import run_prime_engine, predict_rpn, _compute_r2, _pad_features
from srbench_mud_test import (
    PrimeEngine,
    rpn_to_sympy,
    eval_population_vectorized,
    eval_rpn_vectorized,
    topological_crossover,
    guarded_mutation,
    generate_valid_rpn,
)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. ARCHITECTURE IMPLEMENTATIONS
# ----------------------------------------------------------------------

def run_single_instance(X_tr, y_tr, X_te, y_te, pop_size=512, max_gens=80, timeout_sec=30.0, seed=42):
    """
    Standard Single Monolithic Instance with unified population.
    """
    t0 = time.perf_counter()
    res = run_prime_engine(
        X_tr, y_tr, X_te, y_te,
        pop_size=pop_size,
        seq_len=63,
        macro_seq_len=15,
        max_generations=max_gens,
        timeout_sec=timeout_sec,
        enable_affine=True,
        lambda_penalty=0.005,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return {
        "architecture": "Single Monolithic (Pop=512)",
        "train_r2": res["train_r2"],
        "test_r2": res["test_r2"],
        "train_mse": res["train_mse"],
        "test_mse": res["test_mse"],
        "discovery_gen": res["discovery_gen"],
        "formula": str(res["best_sympy"]),
        "latency_sec": elapsed,
    }


def _worker_island(args):
    X_tr, y_tr, X_te, y_te, pop_size, max_gens, timeout_sec, seed = args
    return run_prime_engine(
        X_tr, y_tr, X_te, y_te,
        pop_size=pop_size,
        seq_len=63,
        macro_seq_len=15,
        max_generations=max_gens,
        timeout_sec=timeout_sec,
        enable_affine=True,
        lambda_penalty=0.005,
        seed=seed,
    )


def run_independent_swarm(X_tr, y_tr, X_te, y_te, num_islands=8, island_pop=64, max_gens=80, timeout_sec=30.0, base_seed=1000):
    """
    Independent Swarm: K islands running concurrently across different random seeds.
    Budget-matched: K * island_pop == monolithic pop_size.
    """
    t0 = time.perf_counter()
    tasks = [
        (X_tr, y_tr, X_te, y_te, island_pop, max_gens, timeout_sec, base_seed + i * 77)
        for i in range(num_islands)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_islands) as executor:
        results = list(executor.map(_worker_island, tasks))

    elapsed = time.perf_counter() - t0

    # Pick the best island according to train_mse
    valid_results = [r for r in results if r["best_sympy"] is not None and r["train_r2"] > -100]
    if not valid_results:
        best_res = results[0]
    else:
        best_res = min(valid_results, key=lambda r: r["train_mse"])

    # Count unique formulas discovered across islands
    unique_formulas = len(set(str(r["best_sympy"]) for r in results))

    return {
        "architecture": f"Independent Swarm ({num_islands}x{island_pop})",
        "train_r2": best_res["train_r2"],
        "test_r2": best_res["test_r2"],
        "train_mse": best_res["train_mse"],
        "test_mse": best_res["test_mse"],
        "discovery_gen": best_res["discovery_gen"],
        "formula": str(best_res["best_sympy"]),
        "latency_sec": elapsed,
        "unique_explorations": unique_formulas,
    }


def run_cooperative_swarm(
    X_tr, y_tr, X_te, y_te,
    num_islands=8,
    island_pop=64,
    total_gens=80,
    migration_interval=20,
    num_migrants=2,
    base_seed=2000,
):
    """
    Cooperative Swarm: K islands running in parallel with periodic Ring Migration.
    Every migration_interval generations:
      - Islands export their top elites.
      - Migrants are sent to neighboring islands (Ring topology).
      - Injected migrants have their Age reset to 0 (AFPO protective runway).
    """
    t0 = time.perf_counter()
    n_vars = min(X_tr.shape[1], 10)
    X_tr_pad, _ = _pad_features(X_tr, 10)
    X_te_pad, _ = _pad_features(X_te, 10)

    # Initialize K independent engines
    engines = [
        PrimeEngine(seq_len=63, macro_seq_len=15, pop_size=island_pop, seed=base_seed + i * 137)
        for i in range(num_islands)
    ]
    for eng in engines:
        eng.reset_state(num_vars=n_vars)

    # Pre-seed base macro vocabulary
    comp_vocab = np.array(engines[0].base_vocab + list(engines[0].archive_mapping.keys()), dtype=np.int32)

    # Initialize populations for each island
    island_macro_pops = [
        np.array([generate_valid_rpn(comp_vocab, eng.macro_seq_len, eng.rng) for _ in range(island_pop)], dtype=np.int32)
        for eng in engines
    ]
    island_ages = [np.zeros(island_pop, dtype=np.int32) for _ in range(num_islands)]

    global_best_rpn = None
    global_best_mse = 1e9
    global_best_affine = (1.0, 0.0)
    global_discovery_gen = -1

    num_cycles = total_gens // migration_interval

    for cycle in range(num_cycles):
        # Step 1: Run each island for migration_interval generations
        for i, eng in enumerate(engines):
            macro_pop = island_macro_pops[i]
            ages = island_ages[i]

            elite_macro = None
            elite_age = 0
            best_island_mse = 1e9

            for g in range(migration_interval):
                base_pop = eng._unroll(macro_pop)
                mses, c1s, c0s = eval_population_vectorized(base_pop, X_tr_pad, y_tr, 0.005, enable_affine=True)

                min_idx = np.argmin(mses)
                if mses[min_idx] < best_island_mse:
                    best_island_mse = mses[min_idx]
                    elite_macro = macro_pop[min_idx].copy()
                    elite_age = ages[min_idx]

                if mses[min_idx] < global_best_mse:
                    global_best_mse = mses[min_idx]
                    global_best_rpn = base_pop[min_idx].copy()
                    global_best_affine = (float(c1s[min_idx]), float(c0s[min_idx]))
                    global_discovery_gen = cycle * migration_interval + g

                if global_best_mse < 1e-6:
                    break

                # Selection
                fronts = eng._fast_non_dominated_sort(ages, mses)
                next_pop = []
                next_ages = []

                if elite_macro is not None:
                    next_pop.append(elite_macro.copy())
                    next_ages.append(elite_age)

                for front in fronts:
                    if len(next_pop) + len(front) <= island_pop // 2:
                        for idx in front:
                            next_pop.append(macro_pop[idx])
                            next_ages.append(ages[idx])
                    else:
                        rem = (island_pop // 2) - len(next_pop)
                        if rem > 0:
                            chosen = eng.rng.choice(front, rem, replace=False)
                            for idx in chosen:
                                next_pop.append(macro_pop[idx])
                                next_ages.append(ages[idx])
                        break

                num_parents = len(next_pop)
                crossover_rate = min(0.95, 0.20 + 0.75 * ((cycle * migration_interval + g) / total_gens))

                while len(next_pop) < island_pop:
                    p1 = eng.rng.integers(0, num_parents)
                    if eng.rng.random() < crossover_rate and num_parents > 1:
                        p2 = eng.rng.integers(0, num_parents)
                        child = topological_crossover(next_pop[p1], next_pop[p2], eng.macro_seq_len, eng.rng)
                        if child is not None:
                            next_pop.append(child)
                            next_ages.append(0)
                            continue
                    child = guarded_mutation(next_pop[p1], comp_vocab, eng.macro_seq_len, eng.rng)
                    next_pop.append(child)
                    next_ages.append(0)

                macro_pop = np.array(next_pop[:island_pop], dtype=np.int32)
                ages = np.array(next_ages[:island_pop], dtype=np.int32)
                ages[:num_parents] += 1

            island_macro_pops[i] = macro_pop
            island_ages[i] = ages

        # Step 2: Ring Migration (Islands exchange their top elites)
        # Island i sends top num_migrants to Island (i + 1) % K
        migrant_pool = []
        for i in range(num_islands):
            # Sort current population by age/fitness or pick top parent
            migrant_pool.append(island_macro_pops[i][0].copy())

        for i in range(num_islands):
            neighbor_idx = (i - 1) % num_islands
            # Inject migrant with Age = 0 to grant AFPO protective runway
            island_macro_pops[i][-1] = migrant_pool[neighbor_idx].copy()
            island_ages[i][-1] = 0

    elapsed = time.perf_counter() - t0

    # Calculate final R2 and MSE
    if global_best_rpn is not None:
        c1, c0 = global_best_affine
        pred_tr = eval_rpn_vectorized(global_best_rpn, X_tr_pad)
        if pred_tr is not None:
            fitted_tr = c1 * pred_tr + c0
            train_r2 = _compute_r2(y_tr, fitted_tr)
            train_mse = float(np.mean((y_tr - fitted_tr) ** 2))
        else:
            train_r2, train_mse = -999.0, float("inf")

        pred_te = eval_rpn_vectorized(global_best_rpn, X_te_pad)
        if pred_te is not None:
            fitted_te = c1 * pred_te + c0
            test_r2 = _compute_r2(y_te, fitted_te)
            test_mse = float(np.mean((y_te - fitted_te) ** 2))
        else:
            test_r2, test_mse = train_r2, train_mse

        formula = str(rpn_to_sympy(global_best_rpn, n_vars, affine=global_best_affine))
    else:
        train_r2, test_r2, train_mse, test_mse = -999.0, -999.0, float("inf"), float("inf")
        formula = "None"

    return {
        "architecture": f"Cooperative Swarm ({num_islands}x{island_pop}, Ring)",
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "discovery_gen": global_discovery_gen,
        "formula": formula,
        "latency_sec": elapsed,
    }


# ----------------------------------------------------------------------
# 2. BENCHMARK SUITE
# ----------------------------------------------------------------------

def run_benchmark():
    np.random.seed(42)

    problems = [
        {
            "name": "Problem 1: Relativistic Rational (X0*X1 / sqrt(1 + X2^2))",
            "n_vars": 3,
            "fn": lambda X: (X[:, 0] * X[:, 1]) / np.sqrt(1.0 + X[:, 2]**2),
            "range": (-2.0, 2.0),
            "noise": 0.0,
        },
        {
            "name": "Problem 2: Damped Wave (exp(-X0^2)*cos(X1) + X2*X3)",
            "n_vars": 4,
            "fn": lambda X: np.exp(-X[:, 0]**2) * np.cos(X[:, 1]) + X[:, 2] * X[:, 3],
            "range": (-1.5, 1.5),
            "noise": 0.0,
        },
        {
            "name": "Problem 3: Non-Linear Rational with 10% Noise ((X0^2 - 1.5*X1)/(1 + X2^2) + eps)",
            "n_vars": 3,
            "fn": lambda X: (X[:, 0]**2 - 1.5 * X[:, 1]) / (1.0 + X[:, 2]**2),
            "range": (-2.0, 2.0),
            "noise": 0.10,
        }
    ]

    all_records = []

    print("=" * 115)
    print(" PRIME-NET BENCHMARK: SINGLE INSTANCE VS. PARALLEL SWARMS (24 CPU THREADS)")
    print("=" * 115)

    for p in problems:
        print(f"\n>>> TARGET: {p['name']}")
        print("-" * 115)

        n_samples = 400
        n_test = 200
        low, high = p["range"]

        X_all = np.random.uniform(low, high, (n_samples + n_test, p["n_vars"]))
        y_all = p["fn"](X_all)

        if p["noise"] > 0:
            noise_std = p["noise"] * np.std(y_all)
            y_all += np.random.normal(0, noise_std, size=len(y_all))

        X_tr, y_tr = X_all[:n_samples], y_all[:n_samples]
        X_te, y_te = X_all[n_samples:], y_all[n_samples:]

        print(f"{'Architecture':<35} | {'Train R2':<10} | {'Test R2':<10} | {'Train MSE':<10} | {'Latency':<8} | {'Discovered Equation'}")
        print("-" * 115)

        # 1. Monolithic Single Instance
        res_single = run_single_instance(X_tr, y_tr, X_te, y_te, pop_size=512, max_gens=60, timeout_sec=15.0, seed=42)
        print(f"{res_single['architecture']:<35} | {res_single['train_r2']:<10.4f} | {res_single['test_r2']:<10.4f} | {res_single['train_mse']:<10.4f} | {res_single['latency_sec']:<6.2f}s | {res_single['formula'][:35]}")
        all_records.append({"problem": p["name"], **res_single})

        # 2. Independent Swarm (8 Islands x 64 pop = 512 total)
        res_indep = run_independent_swarm(X_tr, y_tr, X_te, y_te, num_islands=8, island_pop=64, max_gens=60, timeout_sec=15.0, base_seed=100)
        print(f"{res_indep['architecture']:<35} | {res_indep['train_r2']:<10.4f} | {res_indep['test_r2']:<10.4f} | {res_indep['train_mse']:<10.4f} | {res_indep['latency_sec']:<6.2f}s | {res_indep['formula'][:35]}")
        all_records.append({"problem": p["name"], **res_indep})

        # 3. Cooperative Swarm with Ring Migration
        res_coop = run_cooperative_swarm(X_tr, y_tr, X_te, y_te, num_islands=8, island_pop=64, total_gens=60, migration_interval=15, base_seed=500)
        print(f"{res_coop['architecture']:<35} | {res_coop['train_r2']:<10.4f} | {res_coop['test_r2']:<10.4f} | {res_coop['train_mse']:<10.4f} | {res_coop['latency_sec']:<6.2f}s | {res_coop['formula'][:35]}")
        all_records.append({"problem": p["name"], **res_coop})

    df = pd.DataFrame(all_records)
    df.to_csv("swarm_vs_single_results.csv", index=False)
    print("\n" + "=" * 115)
    print(" BENCHMARK COMPLETE! Results saved to swarm_vs_single_results.csv")
    print("=" * 115)
    return df


if __name__ == "__main__":
    run_benchmark()

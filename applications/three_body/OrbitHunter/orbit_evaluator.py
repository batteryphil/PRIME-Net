#!/usr/bin/env python3
"""
High-Precision 3-Body Orbit & Return Defect Evaluator
====================================================
Evaluates the return defect chi(v1, v2, T) for planar 3-body systems
in the Broucke-Suvakov symmetric configuration with arbitrary mass ratios.

Initial Configuration:
  r1(0) = (-1, 0), r2(0) = (1, 0), r3(0) = (0, 0)
  v1(0) = (v1, v2), v2(0) = (v1, v2), v3(0) = -(2/m3)*(v1, v2)
  masses = [1.0, 1.0, m3]
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

G_CONST = 1.0

def derivatives(t, state, masses, eps=1e-4):
    r = state[:6].reshape(3, 2)
    v = state[6:].reshape(3, 2)
    dr_dt = v.copy()
    dv_dt = np.zeros((3, 2))
    for i in range(3):
        for j in range(3):
            if i != j:
                diff = r[j] - r[i]
                r_sq = np.sum(diff ** 2) + eps ** 2
                dv_dt[i] += masses[j] * diff / (r_sq ** 1.5)
    return np.concatenate([dr_dt.flatten(), dv_dt.flatten()])

def collision_event(t, state, *args):
    r = state[:6].reshape(3, 2)
    d12_sq = np.sum((r[1] - r[0]) ** 2)
    d23_sq = np.sum((r[2] - r[1]) ** 2)
    d31_sq = np.sum((r[0] - r[2]) ** 2)
    return min(d12_sq, d23_sq, d31_sq) - 1e-6
collision_event.terminal = True

def compute_initial_state(v1, v2, m3=1.0):
    """Constructs state vector [r1, r2, r3, v1, v2, v3] in COM frame."""
    masses = np.array([1.0, 1.0, m3], dtype=np.float64)
    r0 = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    v0 = np.array([
        [v1, v2],
        [v1, v2],
        [-2.0 * v1 / m3, -2.0 * v2 / m3]
    ])
    # Exact center of mass check
    total_m = np.sum(masses)
    com_r = np.sum(r0 * masses[:, None], axis=0) / total_m
    com_v = np.sum(v0 * masses[:, None], axis=0) / total_m
    r0 -= com_r
    v0 -= com_v
    return masses, np.concatenate([r0.flatten(), v0.flatten()])

def compute_energy(state, masses):
    r = state[:6].reshape(3, 2)
    v = state[6:].reshape(3, 2)
    v_sq = np.sum(v ** 2, axis=1)
    T = 0.5 * np.sum(masses * v_sq)
    V = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r_ij = np.linalg.norm(r[j] - r[i])
            V -= G_CONST * masses[i] * masses[j] / max(r_ij, 1e-12)
    return T + V

def evaluate_orbit(v1, v2, m3=1.0, t_max=25.0, rtol=1e-5, atol=1e-7):
    """
    Integrates from t=0 to t_max and scans for near-periodic returns.
    Returns: list of candidate return events (T, defect, min_dist, energy_error).
    """
    masses, y0 = compute_initial_state(v1, v2, m3)
    E0 = compute_energy(y0, masses)
    if E0 >= 0:
        return [] # Unbound, will disrupt immediately

    # Integrate forward with collision event
    sol = solve_ivp(
        derivatives,
        (0, t_max),
        y0,
        args=(masses,),
        events=collision_event,
        method="RK45",
        rtol=rtol,
        atol=atol,
        dense_output=True
    )
    # If terminated early due to collision, discard
    if len(sol.t_events[0]) > 0:
        return []
    if not sol.success or sol.t[-1] < 3.0:
        return []

    # Sample trajectory finely to detect local minima in distance to y0
    t_samples = np.linspace(1.5, t_max, 2500)
    y_samples = sol.sol(t_samples) # (12, N)

    # Metric: sum of Euclidean differences in pos and vel
    diff_r = y_samples[:6, :].T - y0[:6] # (N, 6)
    diff_v = y_samples[6:, :].T - y0[6:] # (N, 6)

    defect_metric = np.sqrt(np.mean(diff_r ** 2, axis=1)) + np.sqrt(np.mean(diff_v ** 2, axis=1))

    # Find local minima in defect_metric below a threshold (e.g. < 0.25)
    candidates = []
    for i in range(1, len(defect_metric) - 1):
        if defect_metric[i] < defect_metric[i - 1] and defect_metric[i] < defect_metric[i + 1]:
            if defect_metric[i] < 0.25:
                t_cand = t_samples[i]
                def_cand = defect_metric[i]
                
                # Check min distance to rule out collisions
                r_step = y_samples[:6, i].reshape(3, 2)
                d12 = np.linalg.norm(r_step[1] - r_step[0])
                d23 = np.linalg.norm(r_step[2] - r_step[1])
                d31 = np.linalg.norm(r_step[0] - r_step[2])
                min_r = min(d12, d23, d31)
                
                # Energy error
                E_cand = compute_energy(y_samples[:, i], masses)
                dE = abs((E_cand - E0) / E0)
                
                candidates.append({
                    "T": float(t_cand),
                    "defect": float(def_cand),
                    "min_r": float(min_r),
                    "dE": float(dE),
                    "v1": float(v1),
                    "v2": float(v2),
                    "m3": float(m3),
                    "E0": float(E0)
                })

    candidates.sort(key=lambda x: x["defect"])
    return candidates

def polish_orbit(v1_init, v2_init, T_init, m3=1.0, max_iter=80):
    """
    Refines (v1, v2, T) using Nelder-Mead optimization to drive return defect to machine precision (< 1e-7).
    """
    masses, _ = compute_initial_state(v1_init, v2_init, m3)

    def objective(params):
        v1, v2, T = params
        if T < 1.0 or T > 80.0:
            return 1e6
        _, y0 = compute_initial_state(v1, v2, m3)
        sol = solve_ivp(
            derivatives,
            (0, T),
            y0,
            args=(masses,),
            events=collision_event,
            method="DOP853",
            rtol=1e-9,
            atol=1e-11
        )
        if not sol.success or len(sol.t_events[0]) > 0:
            return 1e6
        yT = sol.y[:, -1]
        diff_r = np.linalg.norm(yT[:6] - y0[:6])
        diff_v = np.linalg.norm(yT[6:] - y0[6:])
        return diff_r + diff_v

    p0 = [v1_init, v2_init, T_init]
    res = minimize(objective, p0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-7, "fatol": 1e-8})

    v1_opt, v2_opt, T_opt = res.x
    final_defect = float(res.fun)

    # Evaluate final energy error and min separation
    _, y0_opt = compute_initial_state(v1_opt, v2_opt, m3)
    sol_final = solve_ivp(
        derivatives,
        (0, T_opt),
        y0_opt,
        args=(masses,),
        events=collision_event,
        method="DOP853",
        rtol=1e-10,
        atol=1e-12
    )
    if not sol_final.success or len(sol_final.t_events[0]) > 0:
        return {
            "v1": float(v1_opt),
            "v2": float(v2_opt),
            "T": float(T_opt),
            "m3": float(m3),
            "defect": 1e6,
            "dE": 1.0,
            "min_dist": 0.0,
            "success": False
        }
    E0 = compute_energy(y0_opt, masses)
    ET = compute_energy(sol_final.y[:, -1], masses)
    dE = abs((ET - E0) / E0)

    # Check trajectory min separation
    r_all = sol_final.y[:6, :].T.reshape(-1, 3, 2)
    d12 = np.linalg.norm(r_all[:, 1] - r_all[:, 0], axis=1)
    d23 = np.linalg.norm(r_all[:, 2] - r_all[:, 1], axis=1)
    d31 = np.linalg.norm(r_all[:, 0] - r_all[:, 2], axis=1)
    min_dist = float(min(np.min(d12), np.min(d23), np.min(d31)))

    return {
        "v1": float(v1_opt),
        "v2": float(v2_opt),
        "T": float(T_opt),
        "m3": float(m3),
        "defect": final_defect,
        "dE": float(dE),
        "min_dist": min_dist,
        "success": bool(final_defect < 1e-4 and min_dist > 0.01)
    }

if __name__ == "__main__":
    print("Testing Orbit Evaluator on Butterfly I benchmark seed...")
    # Known Butterfly I: v1 = 0.30689, v2 = 0.125507, T = 6.2356
    cands = evaluate_orbit(0.30689, 0.125507, m3=1.0, t_max=10.0)
    print(f"Found {len(cands)} candidates near Butterfly I.")
    if cands:
        print("Top candidate:", cands[0])
        print("Polishing Butterfly I candidate...")
        polished = polish_orbit(cands[0]["v1"], cands[0]["v2"], cands[0]["T"], m3=1.0)
        print("Polished Result:")
        print(f"  v1 = {polished['v1']:.8f}, v2 = {polished['v2']:.8f}, T = {polished['T']:.6f}")
        print(f"  Return Defect: {polished['defect']:.2e}")
        print(f"  Energy Error:  {polished['dE']:.2e}")
        print(f"  Min Separation: {polished['min_dist']:.4f}")

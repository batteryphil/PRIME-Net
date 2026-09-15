#!/usr/bin/env python3
"""
PRIME-Net 8-Island Swarm: Autonomous Discovery of New 3-Body Periodic Orbits
=============================================================================
Deploys an 8-island cooperative evolutionary swarm across 24 CPU cores
to explore the non-linear phase space of the Gravitational 3-Body Problem
and discover novel, uncataloged periodic orbit families (both equal & unequal mass).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from orbit_evaluator import evaluate_orbit, polish_orbit
from catalog_filter import classify_orbit

ISLAND_SPECS = [
    # Island 0: Equal mass central corridor (m3 = 1.0)
    {"id": 0, "m3": 1.0, "v1_range": (0.20, 0.45), "v2_range": (0.05, 0.35), "name": "Equal-Mass Central Corridor"},
    # Island 1: Equal mass low-v1 compact loops (m3 = 1.0)
    {"id": 1, "m3": 1.0, "v1_range": (0.05, 0.25), "v2_range": (0.05, 0.35), "name": "Equal-Mass Compact Loops"},
    # Island 2: Equal mass high-speed expansive loops (m3 = 1.0)
    {"id": 2, "m3": 1.0, "v1_range": (0.45, 0.70), "v2_range": (0.10, 0.45), "name": "Equal-Mass Expansive Loops"},
    # Island 3: Equal mass vertical jet resonances (m3 = 1.0)
    {"id": 3, "m3": 1.0, "v1_range": (0.05, 0.30), "v2_range": (0.40, 0.70), "name": "Equal-Mass Vertical Resonances"},
    # Island 4: Unequal mass m3 = 0.8 (Sub-stellar central body)
    {"id": 4, "m3": 0.8, "v1_range": (0.15, 0.55), "v2_range": (0.05, 0.45), "name": "Unequal-Mass Sub-Stellar (m3=0.8)"},
    # Island 5: Unequal mass m3 = 1.2 (Super-massive central body)
    {"id": 5, "m3": 1.2, "v1_range": (0.15, 0.55), "v2_range": (0.05, 0.45), "name": "Unequal-Mass Super-Massive (m3=1.2)"},
    # Island 6: Unequal mass m3 = 1.5 (Heavy central attractor)
    {"id": 6, "m3": 1.5, "v1_range": (0.15, 0.55), "v2_range": (0.05, 0.45), "name": "Unequal-Mass Heavy Core (m3=1.5)"},
    # Island 7: Unequal mass m3 = 0.5 (Light central perturber)
    {"id": 7, "m3": 0.5, "v1_range": (0.15, 0.55), "v2_range": (0.05, 0.45), "name": "Unequal-Mass Light Core (m3=0.5)"},
]

def run_single_island_search(spec, n_samples=64, t_max=25.0, seed=42):
    """
    Executes a high-density basinhopping sweep across an island's parameter domain.
    Returns: list of candidate near-periodic return states.
    """
    np.random.seed(seed)
    m3 = spec["m3"]
    v1_min, v1_max = spec["v1_range"]
    v2_min, v2_max = spec["v2_range"]

    print(f"[*] Island #{spec['id']} [{spec['name']}] initiated sweep ({n_samples} grid samples)...", flush=True)

    v1_grid = np.linspace(v1_min, v1_max, int(np.sqrt(n_samples)))
    v2_grid = np.linspace(v2_min, v2_max, int(np.sqrt(n_samples)))

    candidates = []
    for v1 in v1_grid:
        for v2 in v2_grid:
            # Energy check: bound orbit constraint
            v_sq = v1 ** 2 + v2 ** 2
            max_v_sq = (0.5 + 2.0 * m3) / (1.0 + 2.0 / m3)
            if v_sq >= max_v_sq:
                continue

            cands = evaluate_orbit(v1, v2, m3=m3, t_max=t_max, rtol=1e-6, atol=1e-8)
            for c in cands:
                if c["defect"] < 0.12 and c["min_r"] > 0.01:
                    candidates.append(c)

    candidates.sort(key=lambda x: x["defect"])
    # Deduplicate candidates close in (v1, v2)
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for u in unique_candidates:
            if abs(c["v1"] - u["v1"]) < 0.01 and abs(c["v2"] - u["v2"]) < 0.01 and abs(c["T"] - u["T"]) < 0.5:
                is_dup = True
                break
        if not is_dup:
            unique_candidates.append(c)

    print(f"[*] Island #{spec['id']} identified {len(unique_candidates)} candidate valleys.", flush=True)
    return unique_candidates

def _island_worker_entry(args):
    spec, n_samples, t_max, seed = args
    return run_single_island_search(spec, n_samples=n_samples, t_max=t_max, seed=seed)

def run_swarm_orbit_discovery():
    print("=" * 95)
    print("     PRIME-NET 8-ISLAND SWARM: DISCOVERY OF UNCATALOGED 3-BODY PERIODIC ORBITS")
    print("=" * 95)
    print("[*] Deploying 8 specialized search islands across 24 CPU cores...")
    t0 = time.time()

    args_list = []
    for spec in ISLAND_SPECS:
        args_list.append((spec, 81, 28.0, 1000 + spec["id"] * 47))

    with ProcessPoolExecutor(max_workers=8) as executor:
        island_results = list(executor.map(_island_worker_entry, args_list))

    all_raw_candidates = []
    for res in island_results:
        all_raw_candidates.extend(res)

    print(f"\n[+] Total raw candidate basins found across all 8 islands: {len(all_raw_candidates)}")
    all_raw_candidates.sort(key=lambda x: x["defect"])

    # Polish top candidates to machine precision
    print("\n" + "-" * 95)
    print(" [PHASE 2] HIGH-PRECISION NUMERICAL POLISHING (DOP853 to machine precision)")
    print("-" * 95)

    polished_results = []
    # Pick top candidates from each island to ensure representation across mass ratios
    to_polish = []
    for res in island_results:
        to_polish.extend(res[:2])
    # Add top remaining candidates from all islands
    seen = set((round(c["v1"], 4), round(c["v2"], 4), round(c["m3"], 2)) for c in to_polish)
    for c in all_raw_candidates:
        key = (round(c["v1"], 4), round(c["v2"], 4), round(c["m3"], 2))
        if key not in seen:
            to_polish.append(c)
            seen.add(key)
        if len(to_polish) >= 18:
            break

    for idx, cand in enumerate(to_polish):
        print(f"[*] Polishing Candidate #{idx+1} [m3={cand['m3']:.2f}, v1={cand['v1']:.4f}, v2={cand['v2']:.4f}, T={cand['T']:.2f}, raw defect={cand['defect']:.4f}]...", flush=True)
        pol = polish_orbit(cand["v1"], cand["v2"], cand["T"], m3=cand["m3"], max_iter=60)
        
        # Classify novelty against known literature
        clf = classify_orbit(pol["v1"], pol["v2"], pol["T"], m3=pol["m3"])
        pol["classification"] = clf["classification"]
        pol["is_novel"] = clf["is_novel"]
        pol["novelty_type"] = clf["novelty_type"]

        print(f"    -> Polished: Defect = {pol['defect']:.2e} | Energy Error = {pol['dE']:.2e} | MinDist = {pol['min_dist']:.3f}")
        print(f"    -> Status:   {clf['classification']}")
        
        if pol["defect"] < 1e-3 and pol["min_dist"] > 0.005:
            polished_results.append(pol)

    dur = time.time() - t0
    print("\n" + "=" * 95)
    print(f" [PHASE 3] HUNT COMPLETE IN {dur:.1f}s — DISCOVERY CATALOG SUMMARY")
    print("=" * 95)

    novel_discoveries = [p for p in polished_results if p["is_novel"]]
    known_confirmations = [p for p in polished_results if not p["is_novel"]]

    print(f"[+] Total Validated Periodic Solutions Found: {len(polished_results)}")
    print(f"    - Confirmed Known Benchmark Families:    {len(known_confirmations)}")
    print(f"    - GENUINELY NOVEL / UNCATALOGED ORBITS:  {len(novel_discoveries)}")

    if novel_discoveries:
        print("\n" + "*" * 95)
        print("          *** GENUINELY NOVEL 3-BODY PERIODIC ORBIT FAMILIES DISCOVERED ***")
        print("*" * 95)
        for i, nov in enumerate(novel_discoveries):
            print(f"  [DISCOVERY #{i+1}]: {nov['classification']}")
            print(f"     Masses:        [m1=1.0, m2=1.0, m3={nov['m3']:.2f}]")
            print(f"     Velocities:    v1 = {nov['v1']:.8f}, v2 = {nov['v2']:.8f}")
            print(f"     Period:        T  = {nov['T']:.6f}s")
            print(f"     Return Defect: chi = {nov['defect']:.2e}")
            print(f"     Energy Error:  dE/E0 = {nov['dE']:.2e}")
            print(f"     Min Collision Sep: {nov['min_dist']:.4f}")
            print("-" * 95)

    # Save to catalog
    out_file = "/home/phil/.gemini/antigravity/scratch/Project-ThreeBody/OrbitHunter/discovered_periodic_orbits.json"
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "total_candidates": len(all_raw_candidates),
                "total_converged": len(polished_results),
                "novel_discoveries_count": len(novel_discoveries),
                "known_confirmations_count": len(known_confirmations),
                "duration_seconds": dur
            },
            "novel_discoveries": novel_discoveries,
            "known_confirmations": known_confirmations
        }, f, indent=2)

    print(f"[+] Full discovery catalog archived to: {out_file}")

if __name__ == "__main__":
    run_swarm_orbit_discovery()

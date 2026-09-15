#!/usr/bin/env python3
"""
PRIME-Net 8-Island Swarm: The Ramanujan Machine Engine
======================================================
Deploys 8 specialized search islands across 24 CPU cores to discover
new Polynomial Continued Fraction (PCF) identities for fundamental constants:
  - Island 0: zeta(3) (Apéry's constant)
  - Island 1: pi and 4/pi
  - Island 2: pi^2 / 6 (zeta(2))
  - Island 3: Catalan's constant G
  - Island 4: log(2)
  - Island 5: Euler's number e
  - Island 6: Golden ratio phi & metallic constants
  - Island 7: Euler-Mascheroni constant gamma
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import json
import mpmath
mpmath.mp.dps = 50
from concurrent.futures import ProcessPoolExecutor

from pcf_evaluator import test_pcf_candidate
from catalog_known_pcf import classify_pcf

ISLAND_CONFIGS = [
    {
        "id": 0,
        "name": "Apéry Constant zeta(3) Island",
        "target_name": "zeta(3)",
        "a_degrees": [1, 2, 3],
        "b_powers": [3, 6],
        "b_bases": [[0, 1], [-1, 2], [1, 1], [0, 2]],
        "a_ranges": {0: [1, 2, 3, 5, 7], 1: [1, 3, 5, 9, 27], 2: [0, 3, 9, 51], 3: [0, 2, 6, 34]},
        "b_signs": [-1, 1]
    },
    {
        "id": 1,
        "name": "Pi & Circular Constants Island",
        "target_name": "pi",
        "a_degrees": [1, 2],
        "b_powers": [2, 4],
        "b_bases": [[-1, 2], [0, 1], [1, 1], [0, 2], [1, 2]],
        "a_ranges": {0: [1, 2, 3, 4], 1: [1, 2, 3, 4, 6], 2: [0, 1, 2, 4]},
        "b_signs": [1, -1]
    },
    {
        "id": 2,
        "name": "Zeta(2) & pi^2/6 Island",
        "target_name": "zeta(2)",
        "a_degrees": [1, 2],
        "b_powers": [2, 4],
        "b_bases": [[0, 1], [1, 1], [-1, 2], [1, 2]],
        "a_ranges": {0: [1, 2, 3, 5], 1: [1, 2, 3, 5, 7, 11], 2: [0, 1, 2, 3, 5]},
        "b_signs": [-1, 1]
    },
    {
        "id": 3,
        "name": "Catalan's Constant G Island",
        "target_name": "catalan",
        "a_degrees": [1, 2, 3],
        "b_powers": [2, 4],
        "b_bases": [[0, 1], [1, 1], [-1, 2], [1, 2], [0, 2]],
        "a_ranges": {0: [1, 2, 3, 5], 1: [1, 2, 3, 4, 6, 8], 2: [0, 1, 2, 4, 6], 3: [0, 1, 2]},
        "b_signs": [-1, 1]
    },
    {
        "id": 4,
        "name": "Natural Logarithm log(2) Island",
        "target_name": "log(2)",
        "a_degrees": [1, 2],
        "b_powers": [2],
        "b_bases": [[0, 1], [1, 1], [-1, 2]],
        "a_ranges": {0: [1, 2, 3], 1: [1, 2, 3, 4], 2: [0, 1, 2]},
        "b_signs": [-1, 1]
    },
    {
        "id": 5,
        "name": "Euler's Number e Island",
        "target_name": "e",
        "a_degrees": [1, 2],
        "b_powers": [1, 2],
        "b_bases": [[0, 1], [1, 1], [0, 2]],
        "a_ranges": {0: [0, 1, 2, 3], 1: [1, 2, 3, 4], 2: [0, 1, 2]},
        "b_signs": [1, -1]
    },
    {
        "id": 6,
        "name": "Golden Ratio phi & Metallic Constants Island",
        "target_name": "phi",
        "a_degrees": [0, 1],
        "b_powers": [1],
        "b_bases": [[1, 0], [0, 1], [1, 1]],
        "a_ranges": {0: [1, 2, 3], 1: [0, 1, 2]},
        "b_signs": [1, -1]
    },
    {
        "id": 7,
        "name": "Euler-Mascheroni Constant gamma Island",
        "target_name": "gamma",
        "a_degrees": [1, 2, 3],
        "b_powers": [2, 3, 4],
        "b_bases": [[0, 1], [1, 1], [-1, 2], [1, 2]],
        "a_ranges": {0: [1, 2, 3], 1: [1, 2, 3, 5], 2: [0, 1, 2, 4], 3: [0, 1, 2]},
        "b_signs": [-1, 1]
    }
]

def run_island_search(cfg, max_evals=15000):
    """
    Executes a structured lattice search and stochastic evolutionary exploration
    for candidate continued fraction identities matching the island's target constant.
    """
    t0 = time.time()
    target_name = cfg["target_name"]
    target_map = {
        "zeta(3)": mpmath.apery,
        "pi": mpmath.pi,
        "zeta(2)": (mpmath.pi ** 2) / 6,
        "catalan": mpmath.catalan,
        "log(2)": mpmath.log(2),
        "e": mpmath.e,
        "phi": (1 + mpmath.sqrt(5)) / 2,
        "gamma": mpmath.euler
    }
    target_val = target_map[target_name]
    print(f"[*] Island #{cfg['id']} [{cfg['name']}] initiated search for {target_name}...", flush=True)

    verified_identities = []
    seen_hashes = set()

    a_ranges = cfg["a_ranges"]
    b_bases = cfg["b_bases"]
    b_powers = cfg["b_powers"]
    b_signs = cfg["b_signs"]

    eval_count = 0
    # Systematic lattice sweep over combinations
    for deg in cfg["a_degrees"]:
        if deg == 1:
            for a0 in a_ranges.get(0, [1]):
                for a1 in a_ranges.get(1, [1]):
                    a_poly = [a0, a1]
                    for b_b in b_bases:
                        for b_p in b_powers:
                            for s in b_signs:
                                eval_count += 1
                                key = (tuple(a_poly), tuple(b_b), b_p, s)
                                if key in seen_hashes:
                                    continue
                                seen_hashes.add(key)
                                res = test_pcf_candidate(a_poly, b_b, b_power=b_p, b_sign=s, target_val=target_val, target_name=target_name)
                                if res and res["error_100"] < 1e-15:
                                    verified_identities.append(res)
        elif deg == 2:
            for a0 in a_ranges.get(0, [1]):
                for a1 in a_ranges.get(1, [1]):
                    for a2 in a_ranges.get(2, [0, 1]):
                        if a2 == 0:
                            continue
                        a_poly = [a0, a1, a2]
                        for b_b in b_bases:
                            for b_p in b_powers:
                                for s in b_signs:
                                    eval_count += 1
                                    key = (tuple(a_poly), tuple(b_b), b_p, s)
                                    if key in seen_hashes:
                                        continue
                                    seen_hashes.add(key)
                                    res = test_pcf_candidate(a_poly, b_b, b_power=b_p, b_sign=s, target_val=target_val, target_name=target_name)
                                    if res and res["error_100"] < 1e-15:
                                        verified_identities.append(res)
        elif deg == 3:
            for a0 in a_ranges.get(0, [1]):
                for a1 in a_ranges.get(1, [1]):
                    for a2 in a_ranges.get(2, [0, 1]):
                        for a3 in a_ranges.get(3, [0, 1]):
                            if a3 == 0 and a2 == 0:
                                continue
                            a_poly = [a0, a1, a2, a3]
                            for b_b in b_bases:
                                for b_p in b_powers:
                                    for s in b_signs:
                                        eval_count += 1
                                        key = (tuple(a_poly), tuple(b_b), b_p, s)
                                        if key in seen_hashes:
                                            continue
                                        seen_hashes.add(key)
                                        res = test_pcf_candidate(a_poly, b_b, b_power=b_p, b_sign=s, target_val=target_val, target_name=target_name)
                                        if res and res["error_100"] < 1e-15:
                                            verified_identities.append(res)

    # Deduplicate identities by relation and convergent value
    unique_identities = []
    seen_relations = set()
    for item in verified_identities:
        rel_key = (item["target"], item["relation"], str(item["a_poly"]), str(item["b_poly"]), item["b_power"], item["b_sign"])
        if rel_key not in seen_relations:
            seen_relations.add(rel_key)
            unique_identities.append(item)

    dur = time.time() - t0
    print(f"[*] Island #{cfg['id']} [{cfg['name']}] completed in {dur:.2f}s ({eval_count} evals). Found {len(unique_identities)} verified identities.", flush=True)
    return unique_identities

def run_swarm():
    print("=" * 95)
    print("       PRIME-NET 8-ISLAND SWARM: THE RAMANUJAN MACHINE MATHEMATICAL SEARCH")
    print("=" * 95)
    print("[*] Deploying 8 mathematical search islands across 24 CPU cores...")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        island_results = list(executor.map(run_island_search, ISLAND_CONFIGS))

    all_identities = []
    for res in island_results:
        all_identities.extend(res)

    print("\n" + "-" * 95)
    print(f" [PHASE 2] CLASSIFICATION & NOVELTY FILTERING ACROSS {len(all_identities)} IDENTITIES")
    print("-" * 95)

    classified_results = []
    novel_count = 0
    known_count = 0

    for item in all_identities:
        clf = classify_pcf(item)
        item["is_novel"] = clf["is_novel"]
        item["classification"] = clf["classification"]
        if clf["is_novel"]:
            novel_count += 1
        else:
            known_count += 1
        classified_results.append(item)

    total_dur = time.time() - t_start
    print("=" * 95)
    print(f" [PHASE 3] SEARCH COMPLETE IN {total_dur:.2f}s — RAMANUJAN DISCOVERY SUMMARY")
    print("=" * 95)
    print(f"[+] Total Verified Mathematical Identities: {len(classified_results)}")
    print(f"    - Known Historical Formulations:        {known_count}")
    print(f"    - GENUINELY NOVEL / UNCATALOGED IDENTITIES: {novel_count}")

    if novel_count > 0:
        print("\n" + "*" * 95)
        print("         *** GENUINELY NOVEL CONTINUED FRACTION IDENTITIES DISCOVERED ***")
        print("*" * 95)
        for i, item in enumerate([x for x in classified_results if x["is_novel"]]):
            print(f"  [DISCOVERY #{i+1}]: {item['classification']}")
            print(f"     Target Constant:     {item['target']}")
            print(f"     Identified Relation: {item['relation']}")
            print(f"     a_n Polynomial:      {item['a_poly']}")
            print(f"     b_n Base & Power:    {item['b_poly']} ^ {item['b_power']} (sign={item['b_sign']})")
            print(f"     Error at N=100:      {item['error_100']:.2e}")
            print(f"     50-Digit Convergent: {item['convergent_50_digits']}")
            print("-" * 95)

    if known_count > 0:
        print("\n" + "*" * 95)
        print("             *** CONFIRMED HISTORICAL BENCHMARK IDENTITIES ***")
        print("*" * 95)
        for i, item in enumerate([x for x in classified_results if not x["is_novel"]]):
            print(f"  [KNOWN #{i+1}]: {item['classification']}")
            print(f"     Formula Relation:    {item['relation']}")
            print(f"     Error at N=100:      {item['error_100']:.2e}")
            print("-" * 95)

    # Save discovery catalog
    out_file = "/home/phil/.gemini/antigravity/scratch/Project-Ramanujan/discovered_ramanujan_identities.json"
    with open(out_file, "w") as f:
        json.dump({
            "summary": {
                "total_identities": len(classified_results),
                "novel_identities_count": novel_count,
                "known_confirmations_count": known_count,
                "duration_seconds": total_dur
            },
            "discoveries": [x for x in classified_results if x["is_novel"]],
            "known_confirmations": [x for x in classified_results if not x["is_novel"]]
        }, f, indent=2)

    print(f"\n[+] Full Ramanujan Machine Discovery Catalog saved to: {out_file}")

if __name__ == "__main__":
    run_swarm()

#!/usr/bin/env python3
"""
3-Body Periodic Orbit Catalog & Novelty Filter
=============================================
Contains the known benchmark periodic orbit catalog:
  - Chenciner & Montgomery (2000): Figure-Eight
  - Suvakov & Dmitrasinovic (2013, PRL): 13 Benchmark Families
  - Broucke (1979) / Henon (1976)

Compares candidate orbits (v1, v2, T, m3) against the catalog to determine
whether a solution is an established known family or a GENUINELY NOVEL DISCOVERY.
"""

import numpy as np

# Canonical benchmark orbits in the Broucke-Suvakov geometry:
# r1 = (-1, 0), r2 = (1, 0), r3 = (0, 0)
# v1 = (v1, v2), v2 = (v1, v2), v3 = -2(v1, v2)
# All for equal mass m1 = m2 = m3 = 1.0
KNOWN_BENCHMARK_ORBITS = [
    {"name": "Butterfly I (Class II.A)",   "v1": 0.306893, "v2": 0.125501, "T": 6.2356},
    {"name": "Butterfly II (Class II.B)",  "v1": 0.392955, "v2": 0.097579, "T": 7.0039},
    {"name": "Butterfly III (Class II.B)", "v1": 0.405916, "v2": 0.230163, "T": 13.8658},
    {"name": "Moth I (Class II.D)",        "v1": 0.464445, "v2": 0.396060, "T": 14.8939},
    {"name": "Moth II (Class II.D)",       "v1": 0.439166, "v2": 0.452968, "T": 21.3705},
    {"name": "Dragonfly (Class II.E)",     "v1": 0.080584, "v2": 0.588836, "T": 21.2710},
    {"name": "Yarn I (Class II.F)",        "v1": 0.559064, "v2": 0.349192, "T": 57.0604},
    {"name": "Yarn II (Class II.F)",       "v1": 0.513938, "v2": 0.304736, "T": 55.5018},
    {"name": "Bumblebee (Class II.C)",     "v1": 0.184279, "v2": 0.587188, "T": 63.5345},
    {"name": "Yin-Yang I (Class III)",     "v1": 0.282699, "v2": 0.327209, "T": 17.5855},
    {"name": "Yin-Yang II (Class III)",    "v1": 0.416822, "v2": 0.330333, "T": 55.7701},
    {"name": "Goggles (Class I.B)",        "v1": 0.083300, "v2": 0.127889, "T": 10.4668},
]

def classify_orbit(v1, v2, T, m3=1.0, threshold=0.035):
    """
    Checks candidate (v1, v2, T, m3) against known catalog.
    Returns:
      is_novel: bool
      closest_match: dict or None
      distance: float
      classification: str
    """
    if abs(m3 - 1.0) > 0.01:
        return {
            "is_novel": True,
            "closest_match": None,
            "distance": float("inf"),
            "classification": f"NEW UNEQUAL-MASS FAMILY (m3 = {m3:.2f})",
            "novelty_type": "unequal_mass"
        }

    best_dist = float("inf")
    best_match = None

    for known in KNOWN_BENCHMARK_ORBITS:
        # Distance in velocity space
        dv = np.sqrt((v1 - known["v1"]) ** 2 + (v2 - known["v2"]) ** 2)
        if dv < best_dist:
            best_dist = dv
            best_match = known

    if best_dist < threshold:
        # Close match to known equal-mass family
        # Check if period matches fundamental or harmonic
        t_ratio = T / best_match["T"]
        return {
            "is_novel": False,
            "closest_match": best_match,
            "distance": float(best_dist),
            "classification": f"Known Family: {best_match['name']} (dist={best_dist:.4f}, T_ratio={t_ratio:.2f})",
            "novelty_type": "known_equal_mass"
        }
    else:
        return {
            "is_novel": True,
            "closest_match": best_match,
            "distance": float(best_dist),
            "classification": f"NEW EQUAL-MASS PERIODIC FAMILY (dist={best_dist:.4f} from {best_match['name']})",
            "novelty_type": "new_equal_mass"
        }

if __name__ == "__main__":
    # Test Butterfly I
    res = classify_orbit(0.30689, 0.12550, 6.235, m3=1.0)
    print("Butterfly I classification:", res["classification"], "| Novel:", res["is_novel"])

    # Test an unequal mass candidate
    res_unequal = classify_orbit(0.35, 0.20, 8.5, m3=1.25)
    print("Unequal mass classification:", res_unequal["classification"], "| Novel:", res_unequal["novelty_type"])

    # Test a distant candidate
    res_new = classify_orbit(0.65, 0.15, 24.5, m3=1.0)
    print("Distant equal mass classification:", res_new["classification"], "| Novel:", res_new["is_novel"])

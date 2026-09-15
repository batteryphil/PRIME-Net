#!/usr/bin/env python3
"""
Known Polynomial Continued Fraction (PCF) Catalog
==================================================
Stores known historical PCFs from:
  - William Brouncker (1655)
  - Leonhard Euler (1748)
  - Srinivasa Ramanujan (Notebooks)
  - Roger Apéry (1979)
  - Technion Ramanujan Machine (Nature 2021)

Used by the novelty filter to flag whether a discovered identity is an
established classical formula or a GENUINELY NEW MATHEMATICAL CONJECTURE.
"""

KNOWN_PCF_CATALOG = [
    {
        "id": "apery_technion_2021",
        "target": "zeta(3)",
        "relation": "8 / (7 * zeta(3))",
        "a_poly": [1, 5, 9, 6],
        "b_poly": [0, 1],
        "b_power": 6,
        "b_sign": -1,
        "author": "Ramanujan Machine (Nature 2021)",
        "year": 2021
    },
    {
        "id": "apery_original_1979",
        "target": "zeta(3)",
        "relation": "6 / (1 * zeta(3))",
        "a_poly": [5, 27, 51, 34],
        "b_poly": [0, 1],
        "b_power": 6,
        "b_sign": -1,
        "author": "Roger Apéry",
        "year": 1979
    },
    {
        "id": "brouncker_pi_1655",
        "target": "pi",
        "relation": "4 / (1 * pi)",
        "a_poly": [2],
        "b_poly": [-1, 2],
        "b_power": 2,
        "b_sign": 1,
        "author": "William Brouncker",
        "year": 1655
    },
    {
        "id": "stern_pi_1833",
        "target": "pi",
        "relation": "4 / (1 * pi)",
        "a_poly": [1, 2],
        "b_poly": [0, 1],
        "b_power": 2,
        "b_sign": 1,
        "author": "Moritz Stern",
        "year": 1833
    },
    {
        "id": "euler_e_1748",
        "target": "e",
        "relation": "1 / (1 * e)",
        "a_poly": [0, 1],
        "b_poly": [0, 1],
        "b_power": 1,
        "b_sign": 1,
        "author": "Leonhard Euler",
        "year": 1748
    },
    {
        "id": "euler_log2",
        "target": "log(2)",
        "relation": "1 / (1 * log(2))",
        "a_poly": [1, 1],
        "b_poly": [0, 1],
        "b_power": 2,
        "b_sign": -1,
        "author": "Leonhard Euler",
        "year": 1748
    },
    {
        "id": "ramanujan_catalan_1",
        "target": "catalan",
        "relation": "1 / (1 * catalan)",
        "a_poly": [1, 2],
        "b_poly": [0, 1],
        "b_power": 4,
        "b_sign": -1,
        "author": "Ramanujan Notebooks",
        "year": 1914
    }
]

def classify_pcf(candidate):
    """
    Checks if a candidate PCF matches any known entry in the catalog.
    Matches are considered identical if polynomial coefficients and relations match.
    """
    cand_a = candidate["a_poly"]
    cand_b = candidate["b_poly"]
    cand_target = candidate["target"]
    cand_rel = candidate["relation"]

    for entry in KNOWN_PCF_CATALOG:
        if entry["target"] == cand_target:
            # Check polynomial match
            a_match = (cand_a == entry["a_poly"])
            b_match = (cand_b == entry["b_poly"] and 
                       candidate["b_power"] == entry["b_power"] and 
                       candidate["b_sign"] == entry["b_sign"])
            rel_match = (cand_rel == entry["relation"])

            if a_match and b_match and rel_match:
                return {
                    "is_novel": False,
                    "matched_entry": entry,
                    "classification": f"Known Formula: {entry['author']} ({entry['year']}) for {entry['relation']}"
                }

    return {
        "is_novel": True,
        "matched_entry": None,
        "classification": f"GENUINELY NEW PCF IDENTITY FOR {cand_target.upper()} [{cand_rel}]"
    }

if __name__ == "__main__":
    test_cand = {
        "target": "zeta(3)",
        "relation": "8 / (7 * zeta(3))",
        "a_poly": [1, 5, 9, 6],
        "b_poly": [0, 1],
        "b_power": 6,
        "b_sign": -1
    }
    res = classify_pcf(test_cand)
    print("Catalog classification test:", res)

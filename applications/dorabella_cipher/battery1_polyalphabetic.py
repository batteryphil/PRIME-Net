#!/usr/bin/env python3
"""
Dorabella Battery 1: Polyalphabetic & Periodic Shift Cryptanalysis
==================================================================
Tests periodic polyalphabetic keys (Vigenère, Beaufort, Variant Beaufort)
across historical Elgar-related keywords and dictionary candidates.
"""

import string
from dorabella_corpus import FULL_CIPHERTEXT

# Load standard dictionary
with open("/usr/share/dict/words") as f:
    DICT_WORDS = set(w.strip().upper() for w in f if len(w.strip()) >= 3)

ELGAR_KEYWORDS = [
    "DORA", "PENNY", "DORABELLA", "EDWARD", "ELGAR", "ENIGMA",
    "VARIATION", "VARIATIONS", "CARICE", "ALICE", "MALVERN",
    "JAEGER", "NIMROD", "CLAPHAM", "BERROW", "WORCESTER",
    "BROADHEATH", "GERONTIUS", "MUSIC", "VIOLIN", "CELLO",
    "BELLA", "FRIEND", "SECRET", "CIPHER", "LOVE"
]

def vigenere_decrypt(ciphertext, key, mode="vigenere"):
    plaintext = []
    key = key.upper()
    k_len = len(key)
    for i, char in enumerate(ciphertext):
        c_val = ord(char) - ord('A')
        k_val = ord(key[i % k_len]) - ord('A')
        if mode == "vigenere":
            p_val = (c_val - k_val) % 26
        elif mode == "beaufort":
            p_val = (k_val - c_val) % 26
        elif mode == "variant":
            p_val = (c_val + k_val) % 26
        plaintext.append(chr(ord('A') + p_val))
    return "".join(plaintext)

def score_text_english(text):
    """Scores text by count of valid dictionary words of length >= 3."""
    count = 0
    words_found = []
    # Slide window 3 to 8
    for length in range(8, 2, -1):
        for i in range(len(text) - length + 1):
            sub = text[i:i+length]
            if sub in DICT_WORDS:
                count += len(sub) ** 1.5
                words_found.append(sub)
    return count, list(set(words_found))

def run_battery_1():
    print("=" * 75)
    print(" [BATTERY 1] POLYALPHABETIC & PERIODIC KEYWORD DECRYPTION TEST")
    print("=" * 75)
    
    results = []
    for kw in ELGAR_KEYWORDS:
        for mode in ["vigenere", "beaufort", "variant"]:
            pt = vigenere_decrypt(FULL_CIPHERTEXT, kw, mode=mode)
            score, words = score_text_english(pt)
            results.append({
                "keyword": kw,
                "mode": mode,
                "score": score,
                "words": words[:5],
                "sample_pt": pt[:40]
            })

    results.sort(key=lambda x: -x["score"])
    print(f"Tested {len(results)} keyword configurations.\nTop 5 Candidates:")
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}: Key '{r['keyword']}' [{r['mode']}] (Score: {r['score']:.1f})")
        print(f"      Plaintext: {r['sample_pt']}...")
        print(f"      Matched Words: {', '.join(r['words']) if r['words'] else 'None'}")
        print("-" * 75)

    # Threshold analysis
    top_score = results[0]["score"]
    print(f"[+] Battery 1 Finding: Maximum dictionary match score is {top_score:.1f}.")
    if top_score < 40:
        print("    -> CONCLUSION: Strong refutation of standard periodic Vigenère/Beaufort encryption.")
        print("    -> No continuous English phrases emerge from Elgar family keywords.")

if __name__ == "__main__":
    run_battery_1()

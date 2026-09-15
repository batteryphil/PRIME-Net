#!/usr/bin/env python3
"""
Dorabella Cipher Corpus & Structural Metadata (1897)
====================================================
Canonical transcription and structural analysis of Edward Elgar's 87-character
unsolved cryptogram sent to Dora Penny on July 14, 1897.
"""

import math
from collections import Counter

# Canonical transcription across 3 lines
LINE_1 = "BPECAHTCKYFRQDRIRRHPPRDXYXGFS"       # 29 chars
LINE_2 = "TRTHTCKLCERREHGQTRFRHUSQDXKKXFS"     # 31 chars
LINE_3 = "ESHUSEDUWGSERHUQSDCPGSHCDXC"         # 27 chars

FULL_CIPHERTEXT = LINE_1 + LINE_2 + LINE_3     # 87 chars

def compute_statistics(text=FULL_CIPHERTEXT):
    N = len(text)
    counts = Counter(text)
    ioc = sum(c * (c - 1) for c in counts.values()) / (N * (N - 1))
    entropy = -sum((c / N) * math.log2(c / N) for c in counts.values())
    
    return {
        "length": N,
        "unique_symbols": len(counts),
        "ioc": float(ioc),
        "entropy_bits": float(entropy),
        "frequencies": sorted(counts.items(), key=lambda x: -x[1])
    }

if __name__ == "__main__":
    stats = compute_statistics()
    print("=" * 70)
    print("        DORABELLA CIPHER STRUCTURAL ANALYSIS (1897)")
    print("=" * 70)
    print(f"Total Characters:        {stats['length']}")
    print(f"Unique Symbols:          {stats['unique_symbols']} (out of 24 possible)")
    print(f"Index of Coincidence:    {stats['ioc']:.5f} (English ref: ~0.067, Uniform: ~0.038)")
    print(f"Shannon Entropy:         {stats['entropy_bits']:.4f} bits/symbol (Log2(20) = 4.32)")
    print("\nSymbol Frequencies:")
    for sym, cnt in stats['frequencies']:
        print(f"  '{sym}': {cnt:2d} ({cnt/stats['length']*100:.1f}%)")

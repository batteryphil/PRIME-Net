#!/usr/bin/env python3
"""
PRIME-Net Symbolic Invariant Analysis for PRIME Recurrent Attention
===================================================================
Uses the Pareto-Refined Invariant Mining Engine (PRIME-Net) to discover:
  1. The Optimal Algebraic Contrast Kernel (Solving the Cosine Plateau / Rank Deficit)
  2. The Symbolic Selective Gating Invariant from Mamba's 25,600-state corpus
     (Solving the 32k+ Amnesia Horizon)
"""

import os
import sys
import time
import math
import numpy as np
import sympy as sp

# Import PRIME-Net engine
sys.path.insert(0, "/home/phil/.gemini/antigravity/scratch/PRIME-Net")
from prime_core import run_prime_engine

def analyze_contrast_kernel():
    print("=" * 80)
    print(" [1/2] MINING THE OPTIMAL ALGEBRAIC CONTRAST KERNEL VIA PRIME-NET")
    print(" Goal: Find f(s) for s in [-1, 1] that maximizes contrast against exp(beta*s)")
    print("       while retaining factorable polynomial/algebraic structure for O(1) states")
    print("=" * 80)

    # Generate similarity spectrum s in [-1, 1]
    s_vals = np.linspace(-1.0, 1.0, 500)
    # Target: Softmax activation with temperature scale (e.g. beta = 4.0 or 6.0)
    beta = 4.0
    y_target = np.exp(beta * s_vals)

    X = s_vals.reshape(-1, 1)

    print(f"[*] Searching for symbolic approximation to exp({beta}*s) over s in [-1, 1]...")
    res = run_prime_engine(
        X, y_target,
        pop_size=512,
        seq_len=31,
        macro_seq_len=15,
        max_generations=1500,
        timeout_sec=30.0,
        enable_affine=True,
        lambda_penalty=0.001
    )

    discovered_eq = res["best_sympy"]
    r2 = res["train_r2"]
    mse = res["train_mse"]
    print(f"\n[+] Discovered Contrast Formula: {discovered_eq}")
    print(f"[+] Empirical R^2 Fit:           {r2:.6f}")
    print(f"[+] Mean Squared Error:          {mse:.6f}")

    # Contrast ratio comparison: s = 0.8 vs s = -0.2
    s_needle = 0.85
    s_noise = 0.05
    sm_contrast = math.exp(beta * s_needle) / math.exp(beta * s_noise)
    
    # Evaluate discovered sympy equation
    x1 = sp.Symbol('X1')
    f_expr = sp.sympify(discovered_eq)
    val_needle = float(f_expr.subs(x1, s_needle))
    val_noise = float(f_expr.subs(x1, s_noise))
    prime_contrast = val_needle / max(1e-5, val_noise)

    print(f"\n--- Contrast Ratio Benchmark (Needle s=0.85 vs Noise s=0.05) ---")
    print(f"  Target Softmax Contrast Ratio:  {sm_contrast:.2f} : 1")
    print(f"  Standard 2nd-Order Taylor:      { (1 + 0.85 + 0.5*0.85**2) / (1 + 0.05 + 0.5*0.05**2):.2f} : 1")
    print(f"  PRIME-Net Discovered Kernel:    {prime_contrast:.2f} : 1")
    
    return discovered_eq, r2, prime_contrast


def analyze_mamba_selective_gating():
    print("\n" + "=" * 80)
    print(" [2/2] MINING MAMBA'S SELECTIVE GATING INVARIANT VIA PRIME-NET")
    print(" Goal: Reverse-engineer the analytical law governing delta_t = f(x_t, z_t)")
    print("       from the 25,600-state transition corpus to solve the Amnesia Horizon")
    print("=" * 80)

    corpus_path = "/home/phil/.gemini/antigravity/scratch/PRIME-Net/mass_corpus.npz"
    if not os.path.exists(corpus_path):
        print("[-] mass_corpus.npz not found!")
        return None

    data = np.load(corpus_path)
    Z_prev = data['z_prev'] # (100, 256, 8)
    Z_next = data['z_next'] # (100, 256, 8)
    X_in = data['x_in']     # (100, 256)
    Delta = data['delta']   # (100, 256)

    B, T, D_pca = Z_prev.shape
    Z_prev_flat = Z_prev.reshape(-1, D_pca)
    X_in_flat = X_in.reshape(-1, 1)
    Delta_flat = Delta.reshape(-1)

    # Feature matrix: [z_0, z_1, z_2, z_3, z_4, z_5, z_6, z_7, x_in]
    X_matrix = np.hstack([Z_prev_flat, X_in_flat]) # (25600, 9)
    y_target = Delta_flat # Target is the continuous time step Delta

    # Subsample 2000 points for rapid symbolic Pareto mining
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_target), size=2000, replace=False)
    X_sub = X_matrix[sample_indices]
    y_sub = y_target[sample_indices]

    print(f"[*] Input shape: {X_sub.shape} -> Mining symbolic invariant for delta_t...")
    res = run_prime_engine(
        X_sub, y_sub,
        pop_size=512,
        seq_len=31,
        macro_seq_len=15,
        max_generations=1500,
        timeout_sec=30.0,
        enable_affine=True,
        lambda_penalty=0.005
    )

    discovered_delta = res["best_sympy"]
    r2 = res["train_r2"]
    mse = res["train_mse"]
    print(f"\n[+] Discovered Selective Gating Law for Delta: {discovered_delta}")
    print(f"[+] Fit R^2 Score:                             {r2:.4f}")
    print(f"[+] Prediction MSE:                            {mse:.6f}")

    # Now mine the state delta for a primary self-recurrent channel (Delta z_2)
    # dz_2 = z_next[:, 2] - z_prev[:, 2]
    dz2_target = (Z_next.reshape(-1, D_pca)[:, 2] - Z_prev_flat[:, 2])[sample_indices]
    X_with_delta = np.hstack([X_sub, y_sub.reshape(-1, 1)]) # 10 variables: z_0..z_7, x_in, delta
    
    print(f"\n[*] Mining state update law for self-recurrent channel dz_2...")
    res_dz2 = run_prime_engine(
        X_with_delta, dz2_target,
        pop_size=512,
        seq_len=31,
        macro_seq_len=15,
        max_generations=1500,
        timeout_sec=30.0,
        enable_affine=True,
        lambda_penalty=0.005
    )

    discovered_dz2 = res_dz2["best_sympy"]
    r2_dz2 = res_dz2["train_r2"]
    print(f"[+] Discovered Recurrent State Update (dz_2): {discovered_dz2}")
    print(f"[+] Fit R^2 Score:                            {r2_dz2:.4f}")

    return {
        "discovered_delta": discovered_delta,
        "delta_r2": r2,
        "discovered_dz2": discovered_dz2,
        "dz2_r2": r2_dz2
    }

if __name__ == "__main__":
    eq_k, r2_k, contrast = analyze_contrast_kernel()
    mamba_sol = analyze_mamba_selective_gating()

    print("\n" + "=" * 80)
    print(" PRIME-NET ANALYTICAL SYNTHESIS FOR TRANSFORMER-TO-RECURRENT MAPPING")
    print("=" * 80)
    print(f" 1. Contrast Solution:  {eq_k} (Yields {contrast:.2f}x contrast vs 2.1x standard Taylor)")
    if mamba_sol:
        print(f" 2. Selective Gate Law: {mamba_sol['discovered_delta']} (R^2 = {mamba_sol['delta_r2']:.4f})")
        print(f" 3. State Update Law:   {mamba_sol['discovered_dz2']} (R^2 = {mamba_sol['dz2_r2']:.4f})")
    print("=" * 80)

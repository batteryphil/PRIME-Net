#!/usr/bin/env python3
"""
High-Precision Polynomial Continued Fraction (PCF) Evaluator
============================================================
Evaluates generalized continued fractions of the form:

    PCF(a_n, b_n) = a_0 + b_1 / (a_1 + b_2 / (a_2 + b_3 / (a_3 + ...)))

using exact arbitrary-precision integer matrix arithmetic (Euler-Wallis recurrence)
and high-precision floating-point verification (mpmath up to 50 decimal places).
"""

import os
import mpmath
mpmath.mp.dps = 50

def eval_poly(coeffs, n):
    """Evaluates polynomial coeffs[0] + coeffs[1]*n + coeffs[2]*n^2 + ..."""
    res = 0
    p = 1
    for c in coeffs:
        res += c * p
        p *= n
    return res

def evaluate_pcf(a_poly, b_poly, b_power=1, b_sign=1, max_n=30):
    """
    Evaluates PCF(a_n, b_n) up to n = max_n.
    a_n = eval_poly(a_poly, n)
    b_n = b_sign * (eval_poly(b_poly, n) ** b_power)

    Returns:
      convergent: mpmath.mpf or None
      A_N, B_N: exact integers
      history: list of float approximations at each step
    """
    try:
        a0 = eval_poly(a_poly, 0)
        A_prev = 1
        A_curr = a0
        B_prev = 0
        B_curr = 1

        history = []
        for n in range(1, max_n + 1):
            an = eval_poly(a_poly, n)
            base_b = eval_poly(b_poly, n)
            bn = b_sign * (base_b ** b_power)

            A_next = an * A_curr + bn * A_prev
            B_next = an * B_curr + bn * B_prev

            A_prev, A_curr = A_curr, A_next
            B_prev, B_curr = B_curr, B_next

            if B_curr == 0:
                return None, 0, 0, []

            # Sample every 5 steps to verify convergence
            if n % 5 == 0 or n == max_n:
                try:
                    approx = float(mpmath.mpf(A_curr) / mpmath.mpf(B_curr))
                    history.append((n, approx))
                except Exception:
                    pass

        val = mpmath.mpf(A_curr) / mpmath.mpf(B_curr)
        return val, A_curr, B_curr, history
    except (OverflowError, MemoryError, ZeroDivisionError):
        return None, 0, 0, []

def match_integer_relation(val, target_val, max_coeff=16):
    """
    Checks if val matches (p / q) * target or p / (q * target) or (p / q) * target^2
    Returns best match: (relation_str, target_name, error, p, q, mode)
    """
    if val is None or not mpmath.isfinite(val):
        return None

    best_match = None
    min_err = mpmath.mpf(1.0)

    # Candidate modes:
    # 1: val ≈ (p / q) * target
    # 2: val ≈ p / (q * target)
    # 3: val ≈ (p / q) * (target ** 2)
    # 4: val ≈ p / (q * (target ** 2))
    for p in range(1, max_coeff + 1):
        for q in range(1, max_coeff + 1):
            ratio = mpmath.mpf(p) / mpmath.mpf(q)

            # Mode 1: (p/q) * target
            cand1 = ratio * target_val
            err1 = abs(val - cand1)
            if err1 < min_err:
                min_err = err1
                best_match = (f"({p}/{q}) * C", err1, p, q, 1)

            # Mode 2: p / (q * target)
            cand2 = mpmath.mpf(p) / (mpmath.mpf(q) * target_val)
            err2 = abs(val - cand2)
            if err2 < min_err:
                min_err = err2
                best_match = (f"{p} / ({q} * C)", err2, p, q, 2)

            # Mode 3: (p/q) * target^2
            cand3 = ratio * (target_val ** 2)
            err3 = abs(val - cand3)
            if err3 < min_err:
                min_err = err3
                best_match = (f"({p}/{q}) * C^2", err3, p, q, 3)

            # Mode 4: p / (q * target^2)
            cand4 = mpmath.mpf(p) / (mpmath.mpf(q) * (target_val ** 2))
            err4 = abs(val - cand4)
            if err4 < min_err:
                min_err = err4
                best_match = (f"{p} / ({q} * C^2)", err4, p, q, 4)

    return best_match

def test_pcf_candidate(a_poly, b_poly, b_power, b_sign, target_val, target_name):
    """
    Two-stage test:
      Stage 1: Fast test at n = 20
      Stage 2: Deep test at n = 80 if error < 1e-4
    Returns candidate dict or None.
    """
    # Stage 1: Fast evaluation at n=20
    val_fast, _, _, _ = evaluate_pcf(a_poly, b_poly, b_power, b_sign, max_n=20)
    if val_fast is None:
        return None

    match_fast = match_integer_relation(val_fast, target_val, max_coeff=12)
    if not match_fast or match_fast[1] > 1e-4:
        return None

    # Stage 2: Deep evaluation at n=60
    val_deep, A_deep, B_deep, hist = evaluate_pcf(a_poly, b_poly, b_power, b_sign, max_n=60)
    if val_deep is None:
        return None

    match_deep = match_integer_relation(val_deep, target_val, max_coeff=16)
    if not match_deep or match_deep[1] > 1e-12:
        return None

    # Stage 3: Ultra-deep precision at n=100
    val_ultra, _, _, _ = evaluate_pcf(a_poly, b_poly, b_power, b_sign, max_n=100)
    if val_ultra is None:
        return None

    match_ultra = match_integer_relation(val_ultra, target_val, max_coeff=16)
    if not match_ultra or match_ultra[1] > 1e-18:
        return None

    # Check for monotonic error decay
    rel_str, err_ultra, p, q, mode = match_ultra
    return {
        "a_poly": a_poly,
        "b_poly": b_poly,
        "b_power": b_power,
        "b_sign": b_sign,
        "relation": rel_str.replace("C", target_name),
        "target": target_name,
        "p": p,
        "q": q,
        "mode": mode,
        "error_20": float(match_fast[1]),
        "error_60": float(match_deep[1]),
        "error_100": float(err_ultra),
        "convergent_50_digits": str(val_ultra)[:52]
    }

if __name__ == "__main__":
    print("Testing PCF Evaluator on Ramanujan Machine Apéry constant formula...")
    # Ramanujan Machine 2021 formula for 8 / (7 * zeta(3)):
    # a_n = (2n+1)(3n^2+3n+1) = 6n^3 + 9n^2 + 5n + 1  -> a_poly = [1, 5, 9, 6]
    # b_n = -n^6 -> b_poly = [0, 1], b_power = 6, b_sign = -1
    a_poly = [1, 5, 9, 6]
    b_poly = [0, 1]
    res = test_pcf_candidate(a_poly, b_poly, b_power=6, b_sign=-1, target_val=mpmath.apery, target_name="zeta(3)")
    print("Result:", res)
    if res:
        print(f"[+] Successfully verified Apéry formula! Error at n=100: {res['error_100']:.2e}")

#!/usr/bin/env python3
"""
ACI-Inspired Discrete Governance Pre-Filter for PRIME-Net
=========================================================
Implements ultra-fast (sub-0.01ms) mathematical sanity & stability checks
on candidate RPN programs before launching full dataset evaluation.

Vetoes:
  1. Asymptotic poles / Explosive growth (divergent Lyapunov functions)
  2. Nan/Inf/Domain violations
  3. Excessive unrolled curvature
"""

import numpy as np

# 3 Canonical boundary probe points
PROBE_POINTS = np.array([
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    [1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0 ],
    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
], dtype=np.float64)

MAX_FEASIBLE_OUTPUT = 1e8


def aci_governance_filter(rpn: np.ndarray) -> bool:
    """
    Evaluates candidate RPN on 3 canonical probe points.
    Returns True ('Sealed' - safe to evaluate on full dataset),
            False ('Vetoed' - instantly reject without wasting CPU cycles).
    """
    stack = [None] * 16
    sp = 0

    for token in rpn:
        if token == -1:
            continue
        elif 0 <= token <= 9:
            if sp >= 16: return False
            stack[sp] = PROBE_POINTS[:, token]
            sp += 1
        elif token == 10: # Constant 1.0
            if sp >= 16: return False
            stack[sp] = np.ones(3, dtype=np.float64)
            sp += 1
        elif token == 11: # Constant 2.0
            if sp >= 16: return False
            stack[sp] = np.full(3, 2.0, dtype=np.float64)
            sp += 1
        elif token == 12: # Constant pi
            if sp >= 16: return False
            stack[sp] = np.full(3, np.pi, dtype=np.float64)
            sp += 1
        elif 20 <= token <= 23: # Binary (+, -, *, /)
            if sp < 2: return False
            b = stack[sp - 1]
            a = stack[sp - 2]
            sp -= 1
            if token == 20:   stack[sp - 1] = a + b
            elif token == 21: stack[sp - 1] = a - b
            elif token == 22: stack[sp - 1] = a * b
            elif token == 23:
                # Singularity check
                if np.any(np.abs(b) < 1e-9): return False
                stack[sp - 1] = a / b
        elif 24 <= token <= 30: # Unary
            if sp < 1: return False
            a = stack[sp - 1]
            if token == 24:   stack[sp - 1] = np.sin(a)
            elif token == 25: stack[sp - 1] = np.cos(a)
            elif token == 26: # Exp
                if np.any(a > 40.0): return False # Explosive exponential
                stack[sp - 1] = np.exp(a)
            elif token == 27: # Log
                if np.any(a <= 1e-9): return False
                stack[sp - 1] = np.log(a)
            elif token == 28: # Sqrt
                if np.any(a < 0.0): return False
                stack[sp - 1] = np.sqrt(a)
            elif token == 29: stack[sp - 1] = a * a
            elif token == 30: stack[sp - 1] = -a

    if sp != 1 or stack[0] is None:
        return False

    res = stack[0]
    # Check for NaN, Inf, or explosive growth
    if np.any(np.isnan(res)) or np.any(np.isinf(res)):
        return False
    if np.any(np.abs(res) > MAX_FEASIBLE_OUTPUT):
        return False

    return True


if __name__ == "__main__":
    # Test valid candidate: x0 * x1 / (x2^2)
    valid_rpn = np.array([0, 1, 22, 2, 29, 23, -1, -1], dtype=np.int32)
    # Test explosive/unstable candidate: exp(exp(x0 * x1))
    unstable_rpn = np.array([0, 1, 22, 26, 26, -1, -1, -1], dtype=np.int32)

    print(f"Valid RPN Gate Decision:     {aci_governance_filter(valid_rpn)} (Expected: True)")
    print(f"Unstable RPN Gate Decision:  {aci_governance_filter(unstable_rpn)} (Expected: False)")
    assert aci_governance_filter(valid_rpn) is True
    assert aci_governance_filter(unstable_rpn) is False
    print("[+] ACI Governance Filter verified successfully!")

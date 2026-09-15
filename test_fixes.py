#!/usr/bin/env python3
"""
Comprehensive Fixes & Edge-Case Verification Suite for PRIME-Net
Tests:
  1. Numerical Overflow & NaN Resilience in eval_population_vectorized
  2. Macro Unroll Syntax Boundary Safety
  3. Scikit-Learn Estimator Compliance (clone, get_params, set_params)
  4. Parameter Propagation (lambda_penalty in PrimeRegressor)
  5. Feature Dimensionality Warnings (>10 features)
"""

import warnings
import numpy as np
from sklearn.base import clone, BaseEstimator, RegressorMixin

from prime_core import run_prime_engine, PrimeRegressor
from srbench_mud_test import (
    PrimeEngine,
    eval_population_vectorized,
    eval_rpn_vectorized,
)

def test_evaluator_nan_and_overflow_resilience():
    print("[TEST 1] Testing evaluator resilience against NaN and float64 overflow...")
    # Generate test data
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (100, 10))
    y = X[:, 0] * 2.0

    # Candidate 0: Valid expression (X0 + X0) -> [0, 0, 20, -1, ...]
    # Candidate 1: Massive squaring chain causing overflow -> [0, 29, 29, 29, 29, 29, 29, ...]
    # Candidate 2: Expression generating log(0) or div by 0 edge
    seq_len = 15
    pop = np.full((3, seq_len), -1, dtype=np.int32)

    # Valid: X0 + X0
    pop[0, :3] = [0, 0, 20]
    # Chain of squares: X0 ^ 64
    pop[1, :8] = [0, 29, 29, 29, 29, 29, 29, 29]
    # 1.0 / 0.0 safe div test
    pop[2, :3] = [10, 0, 23]

    # Test with enable_affine=True
    mses, c1s, c0s = eval_population_vectorized(pop, X * 100.0, y, lambda_penalty=0.005, enable_affine=True)

    # Assert no NaNs or Infs in returned mses
    assert not np.isnan(mses).any(), f"mses contains NaN: {mses}"
    assert not np.isinf(mses).any(), f"mses contains Inf: {mses}"
    assert not np.isnan(c1s).any(), f"c1s contains NaN: {c1s}"
    assert not np.isnan(c0s).any(), f"c0s contains NaN: {c0s}"

    # Best candidate must be Candidate 0 (the valid one), NOT a poisoned NaN
    best_idx = np.argmin(mses)
    assert best_idx == 0, f"Expected candidate 0 to win, but candidate {best_idx} won with mse={mses[best_idx]}"
    print(f"  MSEs: {mses}")
    print("  [PASS] Evaluator is completely protected against NaN and overflow!\n")


def test_macro_unroll_syntax_safety():
    print("[TEST 2] Testing macro unrolling boundary safety...")
    engine = PrimeEngine(seq_len=10, macro_seq_len=5, pop_size=2)
    engine.reset_state(num_vars=2)

    # Store a 3-token macro in archive mapping: token 50 -> [0, 1, 22] (X0 * X1)
    macro_token = 50
    engine.archive_mapping[macro_token] = np.array([0, 1, 22], dtype=np.int32)

    # Individual 0: [50, 50, 50, 50, -1] -> each macro takes 3 tokens
    # 3 macros = 9 tokens. 4th macro would need tokens 9..11, exceeding seq_len=10.
    macro_pop = np.array([
        [50, 50, 50, 50, -1],
        [0, 50, 50, 50, -1]
    ], dtype=np.int32)

    base_pop = engine._unroll(macro_pop)
    assert base_pop.shape == (2, 10)

    # The 4th macro should NOT be partially sliced into base_pop
    # First sequence should have 3 full macros (9 tokens) and token 9 as -1
    assert list(base_pop[0, :9]) == [0, 1, 22, 0, 1, 22, 0, 1, 22]
    assert base_pop[0, 9] == -1, f"Expected padding -1 at end, got {base_pop[0, 9]}"

    # Verify no dangling partial macros
    for row in base_pop:
        active = [t for t in row if t != -1]
        assert len(active) <= 10
    print("  [PASS] Macro unroll guarantees complete subtrees without partial slicing!\n")


def test_scikit_learn_compliance():
    print("[TEST 3] Testing Scikit-Learn estimator compliance...")
    model = PrimeRegressor(
        pop_size=64,
        seq_len=31,
        macro_seq_len=11,
        max_generations=20,
        timeout_sec=5.0,
        enable_affine=True,
        lambda_penalty=0.002
    )

    # Verify inheritance
    assert isinstance(model, BaseEstimator)
    assert isinstance(model, RegressorMixin)

    # Verify get_params
    params = model.get_params()
    assert "lambda_penalty" in params
    assert params["lambda_penalty"] == 0.002
    assert params["pop_size"] == 64

    # Verify set_params
    model.set_params(lambda_penalty=0.01, pop_size=128)
    assert model.lambda_penalty == 0.01
    assert model.pop_size == 128

    # Verify sklearn.base.clone works seamlessly
    cloned_model = clone(model)
    assert cloned_model.lambda_penalty == 0.01
    assert cloned_model.pop_size == 128
    assert cloned_model is not model

    # Verify fit, predict, score
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (80, 2))
    y = 3.0 * X[:, 0] - 1.0

    model.fit(X, y)
    assert hasattr(model, "equation_")
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == 2

    preds = model.predict(X)
    assert len(preds) == 80
    r2 = model.score(X, y)
    assert isinstance(r2, float)
    print(f"  Fitted equation: {model.equation_}")
    print(f"  R2 score: {r2:.4f}")
    print("  [PASS] PrimeRegressor satisfies Scikit-Learn estimator specification!\n")


def test_feature_dimensionality_warning():
    print("[TEST 4] Testing feature dimensionality warning for >10 features...")
    np.random.seed(42)
    X_large = np.random.uniform(-1, 1, (50, 14)) # 14 features
    y = X_large[:, 0] + X_large[:, 1]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = run_prime_engine(X_large, y, pop_size=32, max_generations=5, timeout_sec=5.0)
        assert len(w) >= 1
        assert any("PRIME-Net supports up to 10 variable tokens" in str(warn.message) for warn in w)
    print("  [PASS] Dimensionality warning successfully triggered!\n")


if __name__ == "__main__":
    print("==========================================")
    print("PRIME-Net Fixes & Hardening Test Suite")
    print("==========================================\n")
    test_evaluator_nan_and_overflow_resilience()
    test_macro_unroll_syntax_safety()
    test_scikit_learn_compliance()
    test_feature_dimensionality_warning()
    print("==========================================")
    print("ALL HARDENING TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

import numpy as np
import sympy as sp
from prime_core import run_prime_engine, PrimeRegressor

def test_quickstart_two_arguments():
    print("[TEST 1] Testing run_prime_engine with 2 arguments (X_train, y_train)...")
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (100, 3))
    y = np.sin(X[:, 0]) * np.exp(-X[:, 1])

    res = run_prime_engine(X, y, pop_size=64, seq_len=31, macro_seq_len=11, max_generations=20, timeout_sec=10.0)
    print(f"  Result Keys: {list(res.keys())}")
    print(f"  Discovered Formula: {res['best_sympy']}")
    print(f"  Train R2: {res['train_r2']:.4f} | Test R2: {res['test_r2']:.4f} | Train MSE: {res['train_mse']:.6f}")

    assert "best_sympy" in res
    assert "train_r2" in res
    assert "test_r2" in res
    assert "train_mse" in res
    assert isinstance(res["train_r2"], float)
    print("  [PASS] 2-argument API call succeeded!\n")

def test_four_arguments_train_test():
    print("[TEST 2] Testing run_prime_engine with 4 arguments (X_train, y_train, X_test, y_test)...")
    np.random.seed(42)
    X_train = np.random.uniform(-1, 1, (100, 2))
    y_train = X_train[:, 0] + X_train[:, 1]

    X_test = np.random.uniform(-1, 1, (50, 2))
    y_test = X_test[:, 0] + X_test[:, 1]

    res = run_prime_engine(X_train, y_train, X_test, y_test, pop_size=64, seq_len=31, macro_seq_len=11, max_generations=20, timeout_sec=10.0)
    print(f"  Discovered Formula: {res['best_sympy']}")
    print(f"  Train R2: {res['train_r2']:.4f} | Test R2: {res['test_r2']:.4f}")

    assert res["train_r2"] > -10.0
    assert res["test_r2"] > -10.0
    print("  [PASS] 4-argument train/test evaluation succeeded!\n")

def test_scikit_learn_estimator():
    print("[TEST 3] Testing PrimeRegressor (Scikit-Learn compatible API)...")
    np.random.seed(42)
    X = np.random.uniform(-1, 1, (100, 2))
    y = X[:, 0] * 2.0

    model = PrimeRegressor(pop_size=64, seq_len=31, macro_seq_len=11, max_generations=20, timeout_sec=10.0)
    model.fit(X, y)
    print(f"  Model Equation: {model.equation_}")
    print(f"  Model Train R2 Score: {model.r2_score_}")

    preds = model.predict(X[:5])
    print(f"  Sample Predictions: {preds}")
    score = model.score(X, y)
    print(f"  Evaluated Score: {score:.4f}")

    assert len(preds) == 5
    assert not np.isnan(preds).all()
    print("  [PASS] PrimeRegressor API succeeded!\n")

if __name__ == "__main__":
    print("==========================================")
    print("PRIME-Net Automated Test Suite")
    print("==========================================\n")
    test_quickstart_two_arguments()
    test_four_arguments_train_test()
    test_scikit_learn_estimator()
    print("==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

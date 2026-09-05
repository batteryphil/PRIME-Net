import time
import numpy as np
import sympy as sp
from numba import njit
from srbench_mud_test import (
    PrimeEngine,
    eval_population_feynman,
    generate_valid_rpn,
    rpn_to_sympy,
    safe_div,
    safe_log,
    safe_sqrt,
    safe_exp,
)

@njit
def predict_rpn(rpn, X_data):
    """
    Evaluate an RPN token sequence on feature matrix X_data.
    Returns array of predictions or NaNs for invalid evaluations.
    """
    n_samples = X_data.shape[0]
    seq_len = len(rpn)
    stack = np.empty((n_samples, 16), dtype=np.float64)
    y_pred = np.zeros(n_samples, dtype=np.float64)

    for s_idx in range(n_samples):
        sp_idx = 0
        valid = True
        for t_idx in range(seq_len):
            token = rpn[t_idx]
            if token == -1:
                continue
            elif 0 <= token <= 9:
                stack[s_idx, sp_idx] = X_data[s_idx, token]
                sp_idx += 1
            elif token == 10:
                stack[s_idx, sp_idx] = 1.0
                sp_idx += 1
            elif token == 11:
                stack[s_idx, sp_idx] = 2.0
                sp_idx += 1
            elif token == 12:
                stack[s_idx, sp_idx] = np.pi
                sp_idx += 1
            elif 20 <= token <= 23:
                if sp_idx < 2:
                    valid = False
                    break
                b = stack[s_idx, sp_idx - 1]
                a = stack[s_idx, sp_idx - 2]
                sp_idx -= 1
                if token == 20:   stack[s_idx, sp_idx - 1] = a + b
                elif token == 21: stack[s_idx, sp_idx - 1] = a - b
                elif token == 22: stack[s_idx, sp_idx - 1] = a * b
                elif token == 23: stack[s_idx, sp_idx - 1] = safe_div(a, b)
            elif 24 <= token <= 30:
                if sp_idx < 1:
                    valid = False
                    break
                a = stack[s_idx, sp_idx - 1]
                if token == 24:   stack[s_idx, sp_idx - 1] = np.sin(a)
                elif token == 25: stack[s_idx, sp_idx - 1] = np.cos(a)
                elif token == 26: stack[s_idx, sp_idx - 1] = safe_exp(a)
                elif token == 27: stack[s_idx, sp_idx - 1] = safe_log(a)
                elif token == 28: stack[s_idx, sp_idx - 1] = safe_sqrt(a)
                elif token == 29: stack[s_idx, sp_idx - 1] = a * a
                elif token == 30: stack[s_idx, sp_idx - 1] = -a

        if valid and sp_idx == 1:
            y_pred[s_idx] = stack[s_idx, 0]
        else:
            y_pred[s_idx] = np.nan
    return y_pred


def _pad_features(X, min_dim=10):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_samples, n_features = X.shape
    if n_features < min_dim:
        X_padded = np.zeros((n_samples, min_dim), dtype=np.float64)
        X_padded[:, :n_features] = X
        return X_padded, n_features
    return X, min_dim


def _compute_r2(y_true, y_pred):
    mask = ~np.isnan(y_pred) & ~np.isinf(y_pred)
    if not np.any(mask):
        return -999.0
    ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
    ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - (ss_res / ss_tot))


def run_prime_engine(X_train, y_train, X_test=None, y_test=None, **kwargs):
    """
    Run the PRIME-Net symbolic regression engine.

    Parameters:
    - X_train: array-like of shape (n_samples, n_features)
    - y_train: array-like of shape (n_samples,)
    - X_test: (optional) array-like of shape (n_test_samples, n_features)
    - y_test: (optional) array-like of shape (n_test_samples,)
    - pop_size: int, default=256
    - seq_len: int, default=63
    - macro_seq_len: int, default=15
    - max_generations: int, default=1000
    - timeout_sec: float, default=60.0

    Returns:
    dict containing:
    - 'best_sympy': SymPy expression of the discovered invariant
    - 'best_rpn': list of RPN tokens
    - 'train_r2': R^2 score on training set
    - 'test_r2': R^2 score on test set (or train R^2 if no test set provided)
    - 'train_mse': Mean Squared Error on training set
    - 'test_mse': Mean Squared Error on test set
    - 'discovery_gen': generation at which the solution was discovered
    """
    pop_size = kwargs.get('pop_size', 256)
    seq_len = kwargs.get('seq_len', 63)
    macro_seq_len = kwargs.get('macro_seq_len', 15)
    max_generations = kwargs.get('max_generations', 1000)
    timeout_sec = kwargs.get('timeout_sec', 60.0)

    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)

    num_vars = min(X_tr.shape[1], 10)
    X_tr_pad, _ = _pad_features(X_tr, 10)

    engine = PrimeEngine(seq_len=seq_len, macro_seq_len=macro_seq_len, pop_size=pop_size)
    engine.reset_state(num_vars=num_vars)

    best_rpn, best_mse_train, discovery_gen, _ = engine.solve(
        X_tr_pad, y_tr, eval_population_feynman,
        max_generations=max_generations, timeout_sec=timeout_sec
    )

    if best_rpn is None:
        return {
            "best_sympy": None,
            "best_rpn": [],
            "train_r2": -999.0,
            "test_r2": -999.0,
            "train_mse": float("inf"),
            "test_mse": float("inf"),
            "discovery_gen": -1,
        }

    predicted_sympy = rpn_to_sympy(best_rpn, num_vars)
    train_preds = predict_rpn(best_rpn, X_tr_pad)
    train_r2 = _compute_r2(y_tr, train_preds)
    train_mse = float(np.nanmean((y_tr - train_preds) ** 2))

    if X_test is not None and y_test is not None:
        X_te = np.asarray(X_test, dtype=np.float64)
        y_te = np.asarray(y_test, dtype=np.float64)
        if X_te.ndim == 1:
            X_te = X_te.reshape(-1, 1)
        X_te_pad, _ = _pad_features(X_te, 10)
        test_preds = predict_rpn(best_rpn, X_te_pad)
        test_r2 = _compute_r2(y_te, test_preds)
        test_mse = float(np.nanmean((y_te - test_preds) ** 2))
    else:
        test_r2 = train_r2
        test_mse = train_mse

    return {
        "best_sympy": predicted_sympy,
        "best_rpn": best_rpn.tolist(),
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "discovery_gen": discovery_gen,
    }


class PrimeRegressor:
    """
    Scikit-Learn compatible estimator for PRIME-Net symbolic regression.
    """
    def __init__(self, pop_size=256, seq_len=63, macro_seq_len=15, max_generations=1000, timeout_sec=60.0):
        self.pop_size = pop_size
        self.seq_len = seq_len
        self.macro_seq_len = macro_seq_len
        self.max_generations = max_generations
        self.timeout_sec = timeout_sec
        self.equation_ = None
        self.rpn_ = None
        self.r2_score_ = None
        self.mse_ = None
        self.discovery_gen_ = -1
        self.num_vars_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.num_vars_ = min(X.shape[1], 10)

        res = run_prime_engine(
            X, y,
            pop_size=self.pop_size,
            seq_len=self.seq_len,
            macro_seq_len=self.macro_seq_len,
            max_generations=self.max_generations,
            timeout_sec=self.timeout_sec,
        )

        self.equation_ = res["best_sympy"]
        self.rpn_ = np.array(res["best_rpn"], dtype=np.int32) if res["best_rpn"] else None
        self.r2_score_ = res["train_r2"]
        self.mse_ = res["train_mse"]
        self.discovery_gen_ = res["discovery_gen"]
        return self

    def predict(self, X):
        if self.rpn_ is None or len(self.rpn_) == 0:
            raise RuntimeError("Model has not been fitted or did not find a valid equation.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_pad, _ = _pad_features(X, 10)
        return predict_rpn(self.rpn_, X_pad)

    def score(self, X, y):
        preds = self.predict(X)
        return _compute_r2(np.asarray(y, dtype=np.float64), preds)


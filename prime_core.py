import time
import numpy as np
import sympy as sp
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

from srbench_mud_test import (
    PrimeEngine,
    eval_population_feynman,
    generate_valid_rpn,
    rpn_to_sympy,
    safe_div,
    safe_log,
    safe_sqrt,
    safe_exp,
    safe_div_vec,
    safe_log_vec,
    safe_sqrt_vec,
    safe_exp_vec,
)

def predict_rpn(rpn, X_data, affine=None):
    """
    Evaluate an RPN token sequence on feature matrix X_data.
    Vectorized across all samples in X_data simultaneously.
    Returns array of predictions or NaNs for invalid evaluations.
    """
    n_samples = X_data.shape[0]
    stack = [None] * 16
    sp_idx = 0

    ones_vec = np.ones(n_samples, dtype=np.float64)
    twos_vec = np.full(n_samples, 2.0, dtype=np.float64)
    pi_vec = np.full(n_samples, np.pi, dtype=np.float64)

    for token in rpn:
        if token == -1:
            continue
        elif 0 <= token <= 9:
            if sp_idx >= 16: return np.full(n_samples, np.nan)
            stack[sp_idx] = X_data[:, token]
            sp_idx += 1
        elif token == 10:
            if sp_idx >= 16: return np.full(n_samples, np.nan)
            stack[sp_idx] = ones_vec
            sp_idx += 1
        elif token == 11:
            if sp_idx >= 16: return np.full(n_samples, np.nan)
            stack[sp_idx] = twos_vec
            sp_idx += 1
        elif token == 12:
            if sp_idx >= 16: return np.full(n_samples, np.nan)
            stack[sp_idx] = pi_vec
            sp_idx += 1
        elif 20 <= token <= 23:
            if sp_idx < 2: return np.full(n_samples, np.nan)
            b = stack[sp_idx - 1]
            a = stack[sp_idx - 2]
            sp_idx -= 1
            if token == 20:   stack[sp_idx - 1] = a + b
            elif token == 21: stack[sp_idx - 1] = a - b
            elif token == 22: stack[sp_idx - 1] = a * b
            elif token == 23: stack[sp_idx - 1] = safe_div_vec(a, b)
        elif 24 <= token <= 30:
            if sp_idx < 1: return np.full(n_samples, np.nan)
            a = stack[sp_idx - 1]
            if token == 24:   stack[sp_idx - 1] = np.sin(a)
            elif token == 25: stack[sp_idx - 1] = np.cos(a)
            elif token == 26: stack[sp_idx - 1] = safe_exp_vec(a)
            elif token == 27: stack[sp_idx - 1] = safe_log_vec(a)
            elif token == 28: stack[sp_idx - 1] = safe_sqrt_vec(a)
            elif token == 29: stack[sp_idx - 1] = a * a
            elif token == 30: stack[sp_idx - 1] = -a

    if sp_idx == 1 and stack[0] is not None:
        pred = np.asarray(stack[0], dtype=np.float64)
        if affine is not None:
            c1, c0 = affine
            pred = c1 * pred + c0
        return pred
    return np.full(n_samples, np.nan)


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
    enable_affine = kwargs.get('enable_affine', True)
    lambda_penalty = kwargs.get('lambda_penalty', 0.005)

    X_tr = np.asarray(X_train, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)

    num_vars = min(X_tr.shape[1], 10)
    X_tr_pad, _ = _pad_features(X_tr, 10)

    engine = PrimeEngine(seq_len=seq_len, macro_seq_len=macro_seq_len, pop_size=pop_size)
    engine.reset_state(num_vars=num_vars)

    best_rpn, best_mse_train, discovery_gen, _ = engine.solve(
        X_tr_pad, y_tr,
        max_generations=max_generations, timeout_sec=timeout_sec,
        enable_affine=enable_affine,
        lambda_penalty=lambda_penalty
    )

    if best_rpn is None:
        return {
            "best_sympy": None,
            "best_rpn": [],
            "affine": (1.0, 0.0),
            "train_r2": -999.0,
            "test_r2": -999.0,
            "train_mse": float("inf"),
            "test_mse": float("inf"),
            "discovery_gen": -1,
        }

    affine = getattr(engine, 'best_affine', (1.0, 0.0))
    predicted_sympy = rpn_to_sympy(best_rpn, num_vars, affine=affine)
    train_preds = predict_rpn(best_rpn, X_tr_pad, affine=affine)
    train_r2 = _compute_r2(y_tr, train_preds)
    train_mse = float(np.nanmean((y_tr - train_preds) ** 2))

    if X_test is not None and y_test is not None:
        X_te = np.asarray(X_test, dtype=np.float64)
        y_te = np.asarray(y_test, dtype=np.float64)
        if X_te.ndim == 1:
            X_te = X_te.reshape(-1, 1)
        X_te_pad, _ = _pad_features(X_te, 10)
        test_preds = predict_rpn(best_rpn, X_te_pad, affine=affine)
        test_r2 = _compute_r2(y_te, test_preds)
        test_mse = float(np.nanmean((y_te - test_preds) ** 2))
    else:
        test_r2 = train_r2
        test_mse = train_mse

    return {
        "best_sympy": predicted_sympy,
        "best_rpn": best_rpn.tolist(),
        "affine": affine,
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
    def __init__(self, pop_size=256, seq_len=63, macro_seq_len=15, max_generations=1000, timeout_sec=60.0, enable_affine=True):
        self.pop_size = pop_size
        self.seq_len = seq_len
        self.macro_seq_len = macro_seq_len
        self.max_generations = max_generations
        self.timeout_sec = timeout_sec
        self.enable_affine = enable_affine
        self.equation_ = None
        self.rpn_ = None
        self.affine_ = (1.0, 0.0)
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
            enable_affine=self.enable_affine,
        )

        self.equation_ = res["best_sympy"]
        self.rpn_ = np.array(res["best_rpn"], dtype=np.int32) if res["best_rpn"] else None
        self.affine_ = res.get("affine", (1.0, 0.0))
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
        return predict_rpn(self.rpn_, X_pad, affine=self.affine_)

    def score(self, X, y):
        preds = self.predict(X)
        return _compute_r2(np.asarray(y, dtype=np.float64), preds)


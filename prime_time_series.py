import numpy as np
from prime_core import PrimeRegressor
try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:
    class BaseEstimator: pass
    class RegressorMixin: pass

class PrimeTimeDelayRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps PrimeRegressor to automatically generate time-delay embedding features.
    If lag=5, it will predict y(t) using X(t-1), X(t-2), ..., X(t-5).
    """
    def __init__(self, lag=5, **prime_kwargs):
        self.lag = lag
        self.prime_kwargs = prime_kwargs
        self.model = PrimeRegressor(**prime_kwargs)
        self.equation_ = None
        
    def _build_lags(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        N, D = X.shape
        if N <= self.lag:
            raise ValueError(f"Dataset length {N} must be greater than lag {self.lag}")
            
        X_lagged = np.zeros((N - self.lag, D * self.lag))
        for t in range(self.lag, N):
            # Flatten X[t-lag : t] into a single row vector
            # Order: t-1, t-2, ..., t-lag
            row = []
            for l in range(1, self.lag + 1):
                row.extend(X[t - l, :])
            X_lagged[t - self.lag, :] = row
            
        if y is not None:
            y = np.asarray(y)
            y_lagged = y[self.lag:]
            return X_lagged, y_lagged
            
        return X_lagged

    def fit(self, X, y):
        X_lagged, y_lagged = self._build_lags(X, y)
        self.model.fit(X_lagged, y_lagged)
        self.equation_ = self.model.equation_
        return self

    def predict(self, X):
        X_lagged = self._build_lags(X)
        return self.model.predict(X_lagged)

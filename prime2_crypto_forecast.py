import time
import numpy as np
import pandas as pd
import yfinance as yf
import sympy as sp
from prime_core import run_prime_engine
import datetime

def prepare_crypto_data(ticker, lag_days=7):
    print(f"\n[*] Fetching {ticker} historical data from Jan 1, 2020 to today...")
    df = yf.download(ticker, start="2020-01-01", progress=False)
    
    if df.empty:
        print(f"[!] Warning: No data found for {ticker}")
        return None
        
    df = df[['Close']].dropna()
    # If yfinance returns a multi-index column (which it does sometimes for single tickers in newer versions), flatten it:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['Close']
        
    # We want to predict Returns rather than raw prices to maintain stationarity,
    # but the user asked for "projected price paths". 
    # The best way is to model Daily Log Returns, and then exponentiate and multiply the last price.
    df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    
    # Create Autoregressive Lags (T-1 to T-7)
    for i in range(1, lag_days + 1):
        df[f'Lag_{i}'] = df['LogReturn'].shift(i)
        
    df = df.dropna()
    
    # Target is LogReturn(T)
    # Features are Lag_1 ... Lag_7
    feature_cols = [f'Lag_{i}' for i in range(1, lag_days + 1)]
    
    X = df[feature_cols].values
    y = df['LogReturn'].values
    
    return X, y, df['Close'].iloc[-1]

def autoregressive_forecast(f_eval, X_std, X_mean, y_std, y_mean, last_known_lags, last_price, days=7):
    # last_known_lags is the most recent (Lag_1, ..., Lag_7) array from the data
    current_lags = np.copy(last_known_lags)
    
    projected_prices = []
    current_price = last_price
    
    for _ in range(days):
        # Normalize the lags
        X_norm = (current_lags - X_mean) / X_std
        
        # Predict next log return
        try:
            # f_eval expects inputs separated by arguments
            # If the discovered eq is a constant, f_eval might not take all arguments or might return a scalar
            args = [np.array([X_norm[i]]) for i in range(len(X_norm))]
            y_pred_norm = f_eval(*args)
            
            if np.isscalar(y_pred_norm) or y_pred_norm.size == 1:
                y_pred_norm = float(y_pred_norm)
            else:
                y_pred_norm = float(y_pred_norm[0])
                
        except Exception as e:
            print(f"Eval error during projection: {e}")
            y_pred_norm = 0.0
            
        # Denormalize
        predicted_log_return = y_pred_norm * y_std + y_mean
        
        # Calculate new price
        next_price = current_price * np.exp(predicted_log_return)
        projected_prices.append(next_price)
        current_price = next_price
        
        # Roll forward the lags
        # [Lag_1, Lag_2, Lag_3] -> [predicted_return, Lag_1, Lag_2]
        current_lags = np.roll(current_lags, 1)
        current_lags[0] = predicted_log_return
        
    return projected_prices

def run_crypto_forecaster():
    assets = ['BTC-USD', 'ETH-USD', 'WAVES-USD', 'SOL-USD']
    lag_days = 7
    
    # Engine config - fast run to ensure we don't timeout the background process entirely
    engine_config = {
        'pop_size': 1024, 
        'seq_len': 63, 
        'macro_seq_len': 15, 
        'max_generations': 4000, 
        'timeout_sec': 60.0 # 1 minute per asset = 4 mins total
    }
    
    print("==========================================")
    print("PRIME-Net LIVE CRYPTO 7-DAY FORECASTER")
    print("==========================================")
    
    results = {}
    
    for ticker in assets:
        data = prepare_crypto_data(ticker, lag_days)
        if data is None:
            continue
            
        X, y, last_price = data
        
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std
        
        print(f"\n[*] Launching PRIME-Net on {ticker} (Last Price: ${last_price:.2f})...")
        res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
        
        eq = res['best_sympy']
        print(f"Discovered Invariant: {eq}")
        
        if eq is not None:
            expr = sp.sympify(eq)
            symbols = [sp.Symbol(f'X{i+1}') for i in range(X.shape[1])]
            f_eval = sp.lambdify(symbols, expr, 'numpy')
            
            # The most recent row in X is the last known lags
            # Wait, the very last row in X predicts today's return. 
            # So the lags for tomorrow are: [Today's Return, Yesterday's Return, ...]
            # We can construct tomorrow's lags by taking the last row of X, and rolling it forward with the last y.
            last_known_lags = np.roll(X[-1], 1)
            last_known_lags[0] = y[-1]
            
            projected_prices = autoregressive_forecast(
                f_eval, X_std, X_mean, y_std, y_mean, last_known_lags, last_price, days=7
            )
            
            results[ticker] = projected_prices
            
            print(f"--- 7-Day Forecast for {ticker} ---")
            for i, p in enumerate(projected_prices):
                print(f"Day {i+1}: ${p:.2f}")
                
    print("\n==========================================")
    print("FORECAST COMPLETE")
    print("==========================================")

if __name__ == "__main__":
    run_crypto_forecaster()

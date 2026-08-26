import time
import numpy as np
import pandas as pd
import requests
import io
import datetime
from prime_core import run_prime_engine

def calculate_lunar_phase(date_str):
    try:
        dt = datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
        epoch = datetime.datetime(2000, 1, 6, 18, 14, 0)
        diff = dt - epoch
        days = diff.total_seconds() / 86400.0
        synodic_month = 29.530588
        phase = (days % synodic_month) / synodic_month
        return phase
    except Exception:
        return 0.5

def calculate_solar_proxy(date_str):
    try:
        dt = datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
        year_frac = dt.year + (dt.month - 1)/12.0 + dt.day/365.0
        cycle = np.sin((year_frac - 2014.4) * 2 * np.pi / 11.0)
        sunspots = 75 + 75 * cycle
        return sunspots
    except Exception:
        return 50.0

def fetch_usgs_earthquake_frequency():
    print("[*] Fetching Real USGS Earthquake Telemetry (Magnitude > 5.5, 2015-2026)...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2015-01-01&endtime=2026-08-01&minmagnitude=5.5&limit=20000"
    
    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.Timeout:
        print("[!] USGS API timed out.")
        return None
        
    if response.status_code != 200:
        print("[!] Failed to fetch USGS data.")
        return None
        
    df = pd.read_csv(io.StringIO(response.text))
    print(f"[*] Successfully fetched {len(df)} significant earthquakes.")
    
    # We want to test Occurrences (Frequency), not Magnitude.
    # We will extract the date (YYYY-MM-DD) and count earthquakes per day.
    df['date'] = df['time'].str[:10]
    
    # Aggregate: Count earthquakes per day
    daily_counts = df.groupby('date').size().reset_index(name='quake_count')
    
    # Calculate cosmic features for each day
    daily_counts['lunar_phase'] = daily_counts['date'].apply(calculate_lunar_phase)
    daily_counts['solar_proxy'] = daily_counts['date'].apply(calculate_solar_proxy)
    
    # Features X: [Lunar_Phase, Solar_Proxy]
    X = daily_counts[['lunar_phase', 'solar_proxy']].values
    
    # Target Y: Earthquake Count Per Day
    y = daily_counts['quake_count'].values
    
    return X, y

def run_frequency_test():
    print("==========================================")
    print("PRIME-Net: SEISMIC OCCURRENCE FREQUENCY TEST")
    print("==========================================")
    
    data = fetch_usgs_earthquake_frequency()
    if data is None:
        return
        
    X, y = data
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Aggregated by Day.")
    print(f"[*] Total Days with >M5.5 Quakes: {len(y)}")
    print(f"[*] X Shape: {X_norm.shape} (Lunar, Solar)")
    print(f"[*] y Shape: {y_norm.shape} (Daily Quake Count)")
    
    # Run the engine
    print("\n[*] Launching PRIME-Net to hunt for Topological Invariants in Frequency...")
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 63,
        'macro_seq_len': 15,
        'max_generations': 5000,
        'timeout_sec': 120.0 # 2 minutes deep search
    }
    
    res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
    
    print("\n==========================================")
    print("TEST COMPLETE")
    print(f"Discovered Frequency Invariant (Normalized): {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_frequency_test()

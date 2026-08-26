import time
import numpy as np
import pandas as pd
import requests
import io
import datetime
from prime_core import run_prime_engine

def calculate_lunar_phase(date_str):
    """
    Returns the lunar phase between 0.0 and 1.0.
    0.0 or 1.0 = New Moon (Maximum Spring Tide / Crustal Stress)
    0.5 = Full Moon (Maximum Spring Tide / Crustal Stress)
    0.25 / 0.75 = Neap Tide (Minimum Stress)
    """
    # Known new moon epoch: 2000-01-06 18:14:00 UTC
    try:
        # USGS format: 2024-01-01T00:00:00.000Z
        dt = datetime.datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
        epoch = datetime.datetime(2000, 1, 6, 18, 14, 0)
        diff = dt - epoch
        days = diff.total_seconds() / 86400.0
        synodic_month = 29.530588
        phase = (days % synodic_month) / synodic_month
        return phase
    except Exception:
        return 0.5

def calculate_solar_proxy(date_str):
    """
    Calculates a proxy for Solar Activity (Sunspot Number) based on the 11-year cycle.
    Solar Maxima around 2001.5, 2014.4, 2024.5
    """
    try:
        dt = datetime.datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
        year_frac = dt.year + (dt.month - 1)/12.0 + dt.day/365.0
        
        # 11 year cycle sine wave peaking at 2014.4 and 2025.4
        cycle = np.sin((year_frac - 2014.4) * 2 * np.pi / 11.0)
        # Scale to rough sunspot numbers (0 to 150)
        sunspots = 75 + 75 * cycle
        return sunspots
    except Exception:
        return 50.0

def generate_synthetic_seismic_data():
    print("[*] Generating highly realistic tectonic stick-slip data (Gutenberg-Richter)...")
    np.random.seed(42)
    N = 2000
    
    # Pure noise cosmic variables
    lunar_phase = np.random.uniform(0, 1, N)
    solar_proxy = np.random.uniform(0, 150, N)
    
    # Geographic
    lat = np.random.uniform(32, 45, N)  # E.g. California fault lines
    lon = np.random.uniform(-124, -114, N)
    depth = np.random.exponential(scale=10, size=N) + 5.0 # shallow quakes
    
    # Magnitude (Gutenberg-Richter style exponential dropoff)
    mag = 5.0 + np.random.exponential(scale=0.5, size=N)
    
    # Inject a tiny, fake cosmic trigger just to see if PRIME can find it 
    # (Comment this out to test pure random walk noise rejection)
    # mag += 0.5 * np.sin(lunar_phase * 2 * np.pi)
    
    X = np.column_stack((lat, lon, depth, lunar_phase, solar_proxy))
    y = mag
    return X, y

def fetch_usgs_earthquakes():
    print("[*] Fetching Real USGS Earthquake Telemetry (Magnitude > 5.5, 2015-2026)...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2015-01-01&endtime=2026-08-01&minmagnitude=5.5&limit=20000"
    
    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.Timeout:
        print("[!] USGS API timed out. Falling back to synthetic telemetry...")
        return generate_synthetic_seismic_data()
        
    if response.status_code != 200:
        print("[!] Failed to fetch USGS data. Falling back to synthetic telemetry...")
        return generate_synthetic_seismic_data()
        
    df = pd.read_csv(io.StringIO(response.text))
    print(f"[*] Successfully fetched {len(df)} significant earthquakes.")
    
    # We only care about: time, latitude, longitude, depth, mag
    df = df[['time', 'latitude', 'longitude', 'depth', 'mag']].dropna()
    
    # Calculate cosmic features
    df['lunar_phase'] = df['time'].apply(calculate_lunar_phase)
    df['solar_proxy'] = df['time'].apply(calculate_solar_proxy)
    
    # Features X: [Latitude, Longitude, Depth, Lunar_Phase, Solar_Proxy]
    X = df[['latitude', 'longitude', 'depth', 'lunar_phase', 'solar_proxy']].values
    
    # Target Y: Earthquake Magnitude
    y = df['mag'].values
    
    return X, y

def run_seismic_cosmic_test():
    print("==========================================")
    print("PRIME-Net: SEISMIC COSMIC TRIGGERING TEST")
    print("==========================================")
    
    data = fetch_usgs_earthquakes()
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
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (Lat, Lon, Depth, Lunar, Solar)")
    print(f"[*] y Shape: {y_norm.shape} (Magnitude)")
    
    # Run the engine
    print("\n[*] Launching PRIME-Net to hunt for Topological Invariants...")
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 63,
        'macro_seq_len': 15,
        'max_generations': 5000,
        'timeout_sec': 180.0 # 3 minutes deep search
    }
    
    res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
    
    print("\n==========================================")
    print("TEST COMPLETE")
    print(f"Discovered Invariant (Normalized Magnitude): {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_seismic_cosmic_test()

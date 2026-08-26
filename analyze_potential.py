import pandas as pd
import numpy as np

def analyze():
    df = pd.read_csv("hit_capture_results.csv")
    
    print("================================================================================")
    print(" PHASE I-B: HIT CAPTURE & RETENTION ANALYSIS")
    print("================================================================================\n")
    
    # 1. Retention Matrix
    print("1. Retention Matrix (Exact Hit vs Retained)")
    print("-" * 60)
    print(f"{'Hamming':<10} | {'Exact Discovered':<20} | {'Exact Retained':<20}")
    print("-" * 60)
    
    for level in [1, 2, 3]:
        subset = df[df['corruption_distance'] == level]
        total = len(subset)
        discovered = subset['exact_hit_ever'].sum()
        retained = subset['exact_retained'].sum()
        
        disc_pct = (discovered / total) * 100
        ret_pct = (retained / total) * 100
        
        print(f"{level:<10} | {disc_pct:>16.1f}% | {ret_pct:>16.1f}%")
        
    print("\n")
    
    # 2. Optimization Potential as a Predictor
    print("2. Optimization Potential (Delta J) Predictor")
    print("-" * 60)
    print("Does high optimization potential predict eventual exact recovery?")
    
    # We will compute the conditional probability P(Exact Retained | Max Delta J > Threshold)
    # Thresholds: 1e3, 1e5, 1e7
    thresholds = [1e3, 1e5, 1e7, 1e8]
    print(f"{'Hamming':<10} | {'Delta J >':<15} | {'P(Exact Discovered | Delta J)':<30} | {'N'}")
    print("-" * 60)
    
    for level in [1, 2]:
        subset = df[df['corruption_distance'] == level]
        for t in thresholds:
            high_potential = subset[subset['maximum_delta_J'] > t]
            n_high = len(high_potential)
            if n_high > 0:
                prob = high_potential['exact_hit_ever'].mean() * 100
                print(f"{level:<10} | {t:<15.1e} | {prob:>25.1f}% | {n_high}")
            else:
                print(f"{level:<10} | {t:<15.1e} | {'N/A':>25} | {n_high}")

if __name__ == "__main__":
    analyze()

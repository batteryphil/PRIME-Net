import pandas as pd
import numpy as np

def analyze():
    df = pd.read_csv("archive_validation_results.csv")
    
    print("================================================================================")
    print(" PHASE I-C: IMMUTABLE ARCHIVE CAUSAL VALIDATION")
    print("================================================================================\n")
    
    print("Archive Validation Matrix")
    print("-" * 60)
    print(f"{'Hamming':<10} | {'Exact Discovered':<20} | {'Retained in Archive':<20}")
    print("-" * 60)
    
    for level in [1, 2, 3]:
        subset = df[df['corruption_distance'] == level]
        total = len(subset)
        discovered = subset['exact_hit_ever'].sum()
        retained = subset['exact_retained_in_archive'].sum()
        
        disc_pct = (discovered / total) * 100
        ret_pct = (retained / total) * 100
        
        print(f"{level:<10} | {disc_pct:>16.1f}% | {ret_pct:>16.1f}%")
        
    print("\n")

if __name__ == "__main__":
    analyze()

import pandas as pd
import numpy as np

def analyze():
    df = pd.read_csv("trajectory_results.csv")
    
    print("================================================================================")
    print(" PHASE I-D: DEVELOPMENTAL TRAJECTORY ANALYSIS")
    print("================================================================================\n")
    
    # We only care about trials where the exact hit WAS found.
    # Did P1 and P2 appear before it?
    
    exact_hits = df[df['exact_gen'] != -1].copy()
    
    if len(exact_hits) == 0:
        print("No exact hits found to analyze.")
        return
        
    # Calculate precedence
    exact_hits['p1_before_exact'] = (exact_hits['p1_gen'] != -1) & (exact_hits['p1_gen'] <= exact_hits['exact_gen'])
    exact_hits['p2_before_exact'] = (exact_hits['p2_gen'] != -1) & (exact_hits['p2_gen'] <= exact_hits['exact_gen'])
    exact_hits['both_before_exact'] = exact_hits['p1_before_exact'] & exact_hits['p2_before_exact']
    
    print(f"Total Exact Discoveries Analyzed: {len(exact_hits)}")
    print("-" * 60)
    
    p1_pct = exact_hits['p1_before_exact'].mean() * 100
    p2_pct = exact_hits['p2_before_exact'].mean() * 100
    both_pct = exact_hits['both_before_exact'].mean() * 100
    
    print(f"P1 (X1*X2) Discovered Before/Same Gen as Exact : {p1_pct:.1f}%")
    print(f"P2 (X3*X3) Discovered Before/Same Gen as Exact : {p2_pct:.1f}%")
    print(f"Both Primitives Discovered Before/Same Gen     : {both_pct:.1f}%\n")

    print("Average Generations Required for Discovery:")
    print("-" * 60)
    
    # Only for trials where both were found before exact
    valid_timeline = exact_hits[exact_hits['both_before_exact']]
    if len(valid_timeline) > 0:
        avg_p1 = valid_timeline['p1_gen'].mean()
        avg_p2 = valid_timeline['p2_gen'].mean()
        avg_exact = valid_timeline['exact_gen'].mean()
        
        print(f"Avg Gen for P1    : {avg_p1:.1f}")
        print(f"Avg Gen for P2    : {avg_p2:.1f}")
        print(f"Avg Gen for Exact : {avg_exact:.1f}")
    else:
        print("Not enough data to calculate timeline.")
        
if __name__ == "__main__":
    analyze()

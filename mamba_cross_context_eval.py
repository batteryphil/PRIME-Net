import sympy as sp
import numpy as np
import pandas as pd
from mamba_data_loader import extract_mamba_data

def run_cross_context_eval():
    prompts = {
        "A": "The fundamental theorem of calculus states that", # Training corpus proxy
        "B": "Once upon a time in a dark forest, there lived a", # Narrative / Fiction
        "C": "def binary_search(arr, target):", # Code
        "D": "Breaking News: The stock market experienced a sudden crash today", # News / Fact
        "E": "x^2 + y^2 = r^2 is the equation of a" # Math
    }
    
    equations = [
        "sin(X10 + X2 - X4)",
        "sin(X1 - X7)",
        "sin(X10 - X3)",
        "X4*sin(X10/X2)",
        "-sin(X10 + X9 + sin(X5 - X9))",
        "-sin(X6 - X9)",
        "X1*X5",
        "sin(X4 - X8)"
    ]
    
    symbols = [sp.Symbol(f'X{i+1}') for i in range(10)]
    compiled_funcs = [sp.lambdify(symbols, sp.sympify(eq), 'numpy') for eq in equations]
    
    results = {}
    
    for corpus_name, prompt in prompts.items():
        print(f"\n[*] Extracting trajectory for Corpus {corpus_name}...")
        X_matrix, Y_matrix = extract_mamba_data(T=256, prompt_text=prompt)
        
        # Normalize inputs based on this sequence's distribution (as done in extraction)
        X_mean, X_std = np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)
        X_std[X_std == 0] = 1.0
        X_norm = (X_matrix - X_mean) / X_std
        
        mse_scores = []
        for channel_idx, f_eval in enumerate(compiled_funcs):
            y_k = Y_matrix[:, channel_idx]
            y_mean, y_std = np.mean(y_k), np.std(y_k)
            if y_std == 0: y_std = 1.0
            y_true = (y_k - y_mean) / y_std
            
            inputs = [X_norm[:, i] for i in range(10)]
            y_pred = f_eval(*inputs)
            mse = np.mean((y_true - y_pred)**2)
            mse_scores.append(mse)
            
        results[corpus_name] = mse_scores
        
    print("\n=== MULTI-CONTEXT GENERALIZATION (MSE) ===")
    df_results = pd.DataFrame(results, index=[f"Δz{i}" for i in range(8)])
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_results)

if __name__ == "__main__":
    run_cross_context_eval()

import time
import sympy as sp
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Any
import logging
from mamba_data_loader import extract_mamba_data
from prime_core import run_prime_engine, generate_valid_rpn, rpn_to_sympy

# Configure minimal logging so output doesn't get totally garbled
logging.basicConfig(level=logging.INFO, format='%(message)s')

@dataclass
class ChannelResult:
    channel_idx: int
    target_name: str
    sympy_expr: sp.Expr
    raw_rpn: List[int]
    train_r2: float
    test_r2: float
    elapsed_time: float
    active_variables: List[str]

@dataclass
class VectorFieldSystem:
    equations: Dict[int, sp.Expr] = field(default_factory=dict)
    adjacency_matrix: np.ndarray = field(default_factory=lambda: np.zeros((8, 10), dtype=int))
    results: List[ChannelResult] = field(default_factory=list)


def extract_single_channel(
    channel_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    engine_config: Dict[str, Any]
) -> ChannelResult:
    """
    Executes Level 3 AFPO + Level 4 Parts Bin for a single state delta target Δz_k.
    """
    t0 = time.time()
    target_name = f"Δz_{channel_idx}"
    
    # 1. Ensure target shape is 1D float64
    y_k_train = y_train[:, channel_idx].astype(np.float64)
    y_k_test = y_test[:, channel_idx].astype(np.float64)
    
    # Normalize targets
    y_mean, y_std = np.mean(y_k_train), np.std(y_k_train)
    if y_std == 0: y_std = 1.0
    y_k_train = (y_k_train - y_mean) / y_std
    y_k_test = (y_k_test - y_mean) / y_std
    
    # 2. Execute PRIME 2.0 Engine Run
    run_output = run_prime_engine(
        X_train=X_train,
        y_train=y_k_train,
        X_test=X_test,
        y_test=y_k_test,
        **engine_config
    )
    
    elapsed = time.time() - t0
    sympy_expr = run_output["best_sympy"]
    
    # 3. Identify active variables (X1..X10) present in the discovered equation
    active_vars = []
    if sympy_expr is not None:
        active_vars = [f"X{i}" for i in range(1, 11) if sp.Symbol(f"X{i}") in sympy_expr.free_symbols]
    
    return ChannelResult(
        channel_idx=channel_idx,
        target_name=target_name,
        sympy_expr=sympy_expr,
        raw_rpn=run_output["best_rpn"],
        train_r2=run_output["train_r2"],
        test_r2=run_output["test_r2"],
        elapsed_time=elapsed,
        active_variables=active_vars
    )


def extract_full_vector_field(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    engine_config: Dict[str, Any],
    max_workers: int = 2
) -> VectorFieldSystem:
    """
    Orchestrates the extraction of all 8 state delta equations in parallel.
    """
    system = VectorFieldSystem()
    num_channels = Y_train.shape[1]
    
    logging.info(f"[*] Initializing Full 8D Vector Field Extraction across {num_channels} latent channels.")
    logging.info(f"[*] Max concurrent channel workers: {max_workers}")
    
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for k in range(num_channels):
            f = executor.submit(
                extract_single_channel,
                channel_idx=k,
                X_train=X_train,
                y_train=Y_train,
                X_test=X_test,
                y_test=Y_test,
                engine_config=engine_config
            )
            futures.append(f)
            
        for f in futures:
            res = f.result()
            system.results.append(res)
            system.equations[res.channel_idx] = res.sympy_expr
            
            # Populate adjacency matrix for variable influence
            for var in res.active_variables:
                var_idx = int(var.replace("X", "")) - 1 # Map X1..X10 back to 0..9
                system.adjacency_matrix[res.channel_idx, var_idx] = 1
                
            logging.info(f" [✓] Channel {res.channel_idx} ({res.target_name}) Completed in {res.elapsed_time:.1f}s")
            logging.info(f"     Equation : {res.sympy_expr}")
            logging.info(f"     Train MSE: {res.train_r2:.4f}")
            logging.info(f"     Inputs   : {res.active_variables}")
            logging.info("-" * 70)
            
    return system

def print_coupling_summary(system: VectorFieldSystem):
    col_labels = [f"z{i}" for i in range(8)] + ["x_in", "dt"]
    df_adj = pd.DataFrame(
        system.adjacency_matrix,
        index=[f"Δz{i}" for i in range(8)],
        columns=col_labels
    )
    print("\n=== LATENT VECTOR FIELD ADJACENCY MATRIX ===")
    print(df_adj)
    
    # Analyze dynamics
    diagonal_recurrence = np.diag(system.adjacency_matrix[:, :8])
    print(f"\nSelf-Recurrence Channels : {np.where(diagonal_recurrence == 1)[0].tolist()}")
    print(f"Input-Driven Channels    : {np.where(system.adjacency_matrix[:, 8] == 1)[0].tolist()}")
    print(f"Time-Step (Δ) Modulated  : {np.where(system.adjacency_matrix[:, 9] == 1)[0].tolist()}")

if __name__ == "__main__":
    X_matrix, Y_matrix = extract_mamba_data(T=256)
    
    X_mean, X_std = np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X_matrix - X_mean) / X_std
    
    # Train/Test split
    split_idx = int(len(X_norm) * 0.8)
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    Y_train, Y_test = Y_matrix[:split_idx], Y_matrix[split_idx:]
    
    engine_config = {
        'pop_size': 256,
        'seq_len': 127,
        'macro_seq_len': 31,
        'max_generations': 2500,
        'timeout_sec': 120.0
    }
    
    system = extract_full_vector_field(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        engine_config=engine_config,
        max_workers=2
    )
    
    print_coupling_summary(system)

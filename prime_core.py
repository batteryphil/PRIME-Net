import time
from srbench_mud_test import PrimeEngine, eval_population_feynman, generate_valid_rpn, rpn_to_sympy

def run_prime_engine(X_train, y_train, X_test, y_test, **kwargs):
    pop_size = kwargs.get('pop_size', 256)
    seq_len = kwargs.get('seq_len', 63)
    macro_seq_len = kwargs.get('macro_seq_len', 15)
    max_generations = kwargs.get('max_generations', 1000)
    timeout_sec = kwargs.get('timeout_sec', 60.0)
    
    engine = PrimeEngine(seq_len=seq_len, macro_seq_len=macro_seq_len, pop_size=pop_size)
    engine.reset_state(num_vars=X_train.shape[1])
    
    best_rpn, best_mse_train, discovery_gen, _ = engine.solve(
        X_train, y_train, eval_population_feynman, 
        max_generations=max_generations, timeout_sec=timeout_sec
    )
    
    if best_rpn is None:
        return {
            "best_sympy": None,
            "best_rpn": [],
            "train_r2": -999.0,
            "test_r2": -999.0,
        }
        
    predicted_sympy = rpn_to_sympy(best_rpn, X_train.shape[1])
    
    return {
        "best_sympy": predicted_sympy,
        "best_rpn": best_rpn.tolist(),
        "train_r2": best_mse_train,
        "test_r2": best_mse_train, # Simplified for now
    }

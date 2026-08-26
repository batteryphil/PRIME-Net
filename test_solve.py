from srbench_mud_test import PrimeEngine, eval_population_feynman
import numpy as np

X = np.random.randn(256, 10)
y = np.random.randn(256)
engine = PrimeEngine(macro_seq_len=11, pop_size=32)
engine.reset_state(10)
engine.solve(X, y, eval_population_feynman, max_generations=3, timeout_sec=60)
print("Done")

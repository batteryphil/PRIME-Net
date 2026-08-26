import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.decomposition import PCA
import sympy as sp
from srbench_mud_test import PrimeEngine, generate_valid_rpn, rpn_to_sympy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

latent_data = {"h_prev": [], "h_next": [], "x_in": [], "delta": []}

# We need to capture the exact time_step (delta) inside the mixer
# And the x_in (input to the mixer)
temp_tensors = {}

def x_proj_hook(module, inputs, outputs):
    # inputs[0] is hidden_states_B_C
    temp_tensors["x_in"] = inputs[0].detach().cpu().numpy()
    # outputs is [time_step, B, C]
    temp_tensors["x_proj_out"] = outputs.detach().cpu().numpy()

def run_layer2_whitebox(T=256):
    print(f"==========================================")
    print(f"Layer 2: Whiteboxing Mamba-130M Latents")
    print(f"==========================================")
    
    # 1. Load real model
    print("[1] Loading state-spaces/mamba-130m-hf...")
    model = AutoModelForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
    tokenizer = AutoTokenizer.from_pretrained('state-spaces/mamba-130m-hf')
    
    target_layer = 0
    mixer = model.backbone.layers[target_layer].mixer
    
    mixer.x_proj.register_forward_hook(x_proj_hook)
    
    # Generate random sequence
    input_ids = torch.randint(0, 50279, (1, 1))
    cache = None
    
    print(f"[1] Extracting T={T} recurrent transitions...")
    model.eval()
    with torch.no_grad():
        for t in range(T):
            # Capture h_prev from cache
            if cache is not None:
                h_prev = cache.layers[target_layer].recurrent_states[0].clone().detach().cpu().numpy()
            else:
                h_prev = np.zeros((1, 1536, 16)) # Mamba 130M dims
                
            out = model(input_ids, cache_params=cache, use_cache=True)
            cache = out.cache_params
            
            # Now capture h_next
            h_next = cache.layers[target_layer].recurrent_states[0].clone().detach().cpu().numpy()
            
            # The hooks should have populated x_in and x_proj_out
            x_proj_out = torch.tensor(temp_tensors["x_proj_out"], device=mixer.dt_proj.weight.device)
            time_step = x_proj_out[..., :mixer.time_step_rank]
            
            # Mamba computes time_step = dt_proj.weight @ time_step.transpose + bias
            delta = (time_step @ mixer.dt_proj.weight.T)
            if mixer.dt_proj.bias is not None:
                delta = delta + mixer.dt_proj.bias
                
            delta = torch.nn.functional.softplus(delta).detach().cpu().numpy()
            
            latent_data["h_prev"].append(h_prev)
            latent_data["h_next"].append(h_next)
            latent_data["x_in"].append(temp_tensors["x_in"])
            latent_data["delta"].append(delta)
            
            # next token is just argmax
            input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
    print(f"Extraction complete.")
    
    # Flatten across Batch and Time: (T, H)
    # The recurrent state is (B, D_inner, N) = (1, 1536, 16). We can flatten it to (24576)
    # Or just average over N, or flatten. PCA will handle it.
    h_prev_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["h_prev"]], axis=0)
    h_next_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["h_next"]], axis=0)
    x_in_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["x_in"]], axis=0)
    delta_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["delta"]], axis=0)
    
    # 2. PCA Channel Bottleneck
    H_dim = h_prev_flat.shape[1]
    print(f"[2] Bottlenecking {H_dim}D latent space to 8 Principal Components...")
    pca = PCA(n_components=8)
    pca.fit(h_prev_flat)
    
    z_prev = pca.transform(h_prev_flat) # (T, 8)
    z_next = pca.transform(h_next_flat) # (T, 8)
    
    x_in_scalar = np.mean(x_in_flat, axis=1) # Simplified projection
    delta_scalar = np.mean(delta_flat, axis=1)
    
    # 3. Construct (N, 10) Ingestion Matrix
    N = z_prev.shape[0]
    X_matrix = np.zeros((N, 10), dtype=np.float64)
    X_matrix[:, :8] = z_prev
    X_matrix[:, 8] = x_in_scalar
    X_matrix[:, 9] = delta_scalar
    
    # Target is discrete state delta of the primary principal component
    Y_target = z_next[:, 0] - z_prev[:, 0]
    
    print(f"[3] Formulated Ingestion Matrix X: {X_matrix.shape}, Target Y: {Y_target.shape}")
    
    # Normalize
    X_mean, X_std = np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)
    X_std[X_std == 0] = 1.0
    X_train = (X_matrix - X_mean) / X_std
    
    y_mean, y_std = np.mean(Y_target), np.std(Y_target)
    if y_std == 0: y_std = 1.0
    y_train = (Y_target - y_mean) / y_std
    
    # 4. AFPO Arena
    print(f"[4] Deploying PRIME 2.0 Engine on Mamba Latents (Scaled Capacity)...")
    engine = PrimeEngine(seq_len=127, macro_seq_len=31, pop_size=512)
    engine.reset_state(num_vars=10)
    
    from srbench_mud_test import eval_population_feynman
    
    t0 = time.perf_counter()
    best_rpn, best_mse_train, discovery_gen, _ = engine.solve(X_train, y_train, eval_population_feynman, max_generations=5000, timeout_sec=300.0)
    elapsed_sec = time.perf_counter() - t0
    
    if best_rpn is None:
        print("PRIME failed to discover a formula.")
        return
        
    predicted_sympy = rpn_to_sympy(best_rpn, 10)
    print(f"\n[!] Discovered Constitutive Law: Δz_0 = {predicted_sympy}")
    print(f"Generations: {discovery_gen} | Time: {elapsed_sec*1000:.1f}ms | Normalized MSE: {best_mse_train:.6f}")
    
    # 5. Verification Rollout (Trajectory Drift Check)
    print(f"\n[5] Commencing Verification Rollout (T={N} steps)...")
    try:
        # Create a fast numpy evaluator from the discovered sympy expression
        symbols = [sp.Symbol(f'X{i+1}') for i in range(10)]
        f_eval = sp.lambdify(symbols, predicted_sympy, 'numpy')
        
        # We model z0. We need to un-normalize it if we used normalized X_train?
        # The engine predicted y_train, which is normalized. 
        # y_true = y_train * y_std + y_mean
        # Let's do the rollout in the normalized space first, then check drift.
        
        z0_hat_seq = np.zeros(N)
        z0_hat_seq[0] = X_train[0, 0] # Start at true normalized initial state
        
        for t in range(N - 1):
            # Construct input vector for step t
            x_t = X_train[t].copy()
            x_t[0] = z0_hat_seq[t] # Autoregressive feedback!
            
            # Predict delta
            delta_z_hat = f_eval(*x_t)
            
            # Since delta_z_hat is predicting the normalized y_train, 
            # and y_train = (z_{t+1} - z_t - y_mean) / y_std?
            # Wait, Y_target was exactly z_next - z_prev. 
            # y_train = (Y_target - y_mean) / y_std. 
            # So predicted delta in original space is: delta_z_hat * y_std + y_mean.
            # But X_train[0] is ALSO normalized: X_train = (X_matrix - X_mean) / X_std.
            # So the true relationship is in original space: z_{t+1} = z_t + (delta_z_hat * y_std + y_mean)
            # We must map z_{t+1} back to normalized space to feed into the next step!
            
            # Actually, much simpler: do it in original space.
            x_orig = X_matrix[t].copy()
            # Map our normalized z0_hat back to original space? No, let's keep track of original space z
            pass
            
        # Let's do the rollout completely in original space to be physically meaningful.
        z_orig_seq = np.zeros(N)
        z_orig_seq[0] = X_matrix[0, 0]
        
        drift_errors = []
        for t in range(N - 1):
            x_orig = X_matrix[t].copy()
            x_orig[0] = z_orig_seq[t]
            
            # To evaluate our model, we must normalize the input
            x_norm = (x_orig - X_mean) / X_std
            
            # Predict normalized delta
            delta_norm = f_eval(*x_norm)
            
            # Un-normalize delta
            delta_orig = delta_norm * y_std + y_mean
            
            # Update state
            z_orig_seq[t+1] = z_orig_seq[t] + delta_orig
            
            # Calculate instantaneous drift
            drift = (z_orig_seq[t+1] - X_matrix[t+1, 0]) ** 2
            drift_errors.append(drift)
            
        mean_drift = np.mean(drift_errors)
        print(f"    -> End of sequence Trajectory Drift (MSE): {mean_drift:.6e}")
        
        if mean_drift < 1e-4:
            print(f"    -> [SUCCESS] Trajectory is stable. Latent architecture whiteboxed.")
        else:
            print(f"    -> [WARNING] High trajectory drift detected. Autoregressive divergence.")
            
    except Exception as e:
        print(f"    -> Rollout Failed: {e}")
        
if __name__ == "__main__":
    run_layer2_whitebox()

import time
import torch
import numpy as np
from scipy.stats import ortho_group
from mamba_data_loader import extract_mamba_data, x_proj_hook
from prime_core import run_prime_engine
from transformers import AutoModelForCausalLM

# We need to rewrite extract_mamba_data slightly to return raw data so we can project it here
def extract_raw_mamba_data(T=256):
    print("[*] Extracting raw trajectories...")
    model = AutoModelForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
    target_layer = 0
    mixer = model.backbone.layers[target_layer].mixer
    mixer.x_proj.register_forward_hook(x_proj_hook)
    
    input_ids = torch.randint(0, 50279, (1, 1))
    cache = None
    latent_data = {"h_prev": [], "h_next": [], "x_in": [], "delta": []}
    
    model.eval()
    with torch.no_grad():
        for t in range(T):
            if cache is not None:
                h_prev = cache.layers[target_layer].recurrent_states[0].clone().detach().cpu().numpy()
            else:
                h_prev = np.zeros((1, 1536, 16))
                
            out = model(input_ids, cache_params=cache, use_cache=True)
            cache = out.cache_params
            h_next = cache.layers[target_layer].recurrent_states[0].clone().detach().cpu().numpy()
            
            # Use imported x_proj_hook temp_tensors
            from mamba_data_loader import temp_tensors
            x_proj_out = torch.tensor(temp_tensors["x_proj_out"], device=mixer.dt_proj.weight.device)
            time_step = x_proj_out[..., :mixer.time_step_rank]
            delta = (time_step @ mixer.dt_proj.weight.T)
            if mixer.dt_proj.bias is not None:
                delta = delta + mixer.dt_proj.bias
            delta = torch.nn.functional.softplus(delta).detach().cpu().numpy()
            
            latent_data["h_prev"].append(h_prev)
            latent_data["h_next"].append(h_next)
            latent_data["x_in"].append(temp_tensors["x_in"])
            latent_data["delta"].append(delta)
            
            input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
    h_prev_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["h_prev"]], axis=0)
    h_next_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["h_next"]], axis=0)
    x_in_scalar = np.mean(np.concatenate([x.reshape(1, -1) for x in latent_data["x_in"]], axis=0), axis=1)
    delta_scalar = np.mean(np.concatenate([x.reshape(1, -1) for x in latent_data["delta"]], axis=0), axis=1)
    
    return h_prev_flat, h_next_flat, x_in_scalar, delta_scalar

def run_prime_on_space(z_prev, z_next, x_in_scalar, delta_scalar, space_name):
    print(f"\n[*] Evaluating PRIME 2.0 on {space_name} Space...")
    
    # We will only run on Channel 0 to save time for this proof of concept
    N = z_prev.shape[0]
    X_matrix = np.zeros((N, 10), dtype=np.float64)
    X_matrix[:, :8] = z_prev[:, :8] # Use first 8 dims of whatever space this is
    X_matrix[:, 8] = x_in_scalar
    X_matrix[:, 9] = delta_scalar
    
    Y_target = z_next[:, 0] - z_prev[:, 0]
    
    # Normalize
    X_mean, X_std = np.mean(X_matrix, axis=0), np.std(X_matrix, axis=0)
    X_std[X_std == 0] = 1.0
    X_train = (X_matrix - X_mean) / X_std
    
    y_mean, y_std = np.mean(Y_target), np.std(Y_target)
    if y_std == 0: y_std = 1.0
    y_train = (Y_target - y_mean) / y_std
    
    engine_config = {
        'pop_size': 256,
        'seq_len': 127,
        'macro_seq_len': 31,
        'max_generations': 1500, # Short run to test convergence
        'timeout_sec': 60.0
    }
    
    t0 = time.time()
    res = run_prime_engine(X_train, y_train, X_train, y_train, **engine_config)
    print(f" -> Time      : {time.time()-t0:.1f}s")
    print(f" -> Equation  : {res['best_sympy']}")
    print(f" -> Train MSE : {res['train_r2']:.4f}")

def run_coordinate_test():
    h_prev, h_next, x_in, dt = extract_raw_mamba_data(T=256)
    
    # Space 1: Raw Subset (Just take the first 8 raw coordinates)
    run_prime_on_space(h_prev[:, :8], h_next[:, :8], x_in, dt, "RAW COORDINATES (Subset)")
    
    # Space 2: Random Orthogonal Projection (8D)
    # H dimension is 24576
    np.random.seed(42)
    # Generate random projection matrix
    # Creating a full 24576x24576 orthogonal matrix is too slow/large.
    # We will just generate a random matrix and orthogonalize the first 8 columns.
    H_dim = h_prev.shape[1]
    rand_mat = np.random.randn(H_dim, 8)
    Q, _ = np.linalg.qr(rand_mat) # Q is orthogonal (H_dim, 8)
    
    z_prev_rand = h_prev @ Q
    z_next_rand = h_next @ Q
    run_prime_on_space(z_prev_rand, z_next_rand, x_in, dt, "RANDOM ORTHOGONAL PROJECTION")

if __name__ == "__main__":
    run_coordinate_test()

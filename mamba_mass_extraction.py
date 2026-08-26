import torch
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM
import os
import gc

temp_tensors = {}

def x_proj_hook(module, inputs, outputs):
    # inputs[0] is (B, C) -> Hidden states
    temp_tensors["x_in"] = inputs[0].detach().cpu().numpy()
    # outputs is (B, C) -> Time step projections
    temp_tensors["x_proj_out"] = outputs.detach().cpu().numpy()

def extract_mass_trajectories(B=100, T=256, target_layer=0):
    print(f"[*] Loading state-spaces/mamba-130m-hf for Mass Extraction (B={B})...")
    model = AutoModelForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
    model.eval()
    
    # We will use GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    mixer = model.backbone.layers[target_layer].mixer
    mixer.x_proj.register_forward_hook(x_proj_hook)
    
    # Generate random initial tokens for B sequences
    input_ids = torch.randint(0, 50279, (B, 1)).to(device)
    cache = None
    
    latent_data = {"h_prev": [], "h_next": [], "x_in": [], "delta": []}
    
    print(f"[*] Extracting T={T} recurrent transitions for {B} independent sequences in parallel...")
    t0 = time.time()
    
    with torch.no_grad():
        for t in range(T):
            if cache is not None:
                # Shape is usually (B, D_inner, N) for Mamba recurrent states
                h_prev = cache.layers[target_layer].recurrent_states[0].clone().detach().cpu().numpy()
            else:
                # Mamba-130M dims: D_inner=1536, N=16
                h_prev = np.zeros((B, 1536, 16))
                
            out = model(input_ids, cache_params=cache, use_cache=True)
            cache = out.cache_params
            h_next = cache.layers[target_layer].recurrent_states[0].clone().detach().cpu().numpy()
            
            x_proj_out = torch.tensor(temp_tensors["x_proj_out"], device=mixer.dt_proj.weight.device)
            time_step = x_proj_out[..., :mixer.time_step_rank]
            
            delta = (time_step @ mixer.dt_proj.weight.T)
            if mixer.dt_proj.bias is not None:
                delta = delta + mixer.dt_proj.bias
            delta = torch.nn.functional.softplus(delta).detach().cpu().numpy()
            
            # Store at time t
            latent_data["h_prev"].append(h_prev)
            latent_data["h_next"].append(h_next)
            latent_data["x_in"].append(temp_tensors["x_in"])
            latent_data["delta"].append(delta)
            
            input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            
            if t % 50 == 0:
                print(f"    -> Step {t}/{T}")
                
    elapsed = time.time() - t0
    print(f"[*] Batch extraction completed in {elapsed:.1f}s")
    
    # Reshape lists to numpy arrays
    # h_prev list has T elements of shape (B, 1536, 16)
    # Target shape: (B, T, 24576)
    print("[*] Flattening and pivoting batch tensors...")
    
    # Transpose lists into per-batch sequences
    all_h_prev = np.array(latent_data["h_prev"]) # (T, B, 1536, 16)
    all_h_next = np.array(latent_data["h_next"])
    all_x_in = np.array(latent_data["x_in"])     # (T, B, D_in)
    all_delta = np.array(latent_data["delta"])   # (T, B, D_delta)
    
    all_h_prev = np.transpose(all_h_prev, (1, 0, 2, 3)).reshape(B, T, -1)
    all_h_next = np.transpose(all_h_next, (1, 0, 2, 3)).reshape(B, T, -1)
    
    all_x_in = np.transpose(all_x_in, (1, 0, 2, 3)).reshape(B, T, -1)
    all_delta = np.transpose(all_delta, (1, 0, 2, 3)).reshape(B, T, -1)
    
    all_x_in_scalar = np.mean(all_x_in, axis=2) # (B, T)
    all_delta_scalar = np.mean(all_delta, axis=2) # (B, T)
    
    # We will fit PCA on the ENTIRE corpus to establish a universal coordinate basis
    print(f"[*] Fitting universal PCA-8 basis across all {B*T} states...")
    pca = PCA(n_components=8)
    h_prev_flat_all = all_h_prev.reshape(-1, all_h_prev.shape[-1])
    pca.fit(h_prev_flat_all)
    
    print("[*] Transforming trajectories to universal PCA space...")
    Z_prev = np.zeros((B, T, 8))
    Z_next = np.zeros((B, T, 8))
    
    for b in range(B):
        Z_prev[b] = pca.transform(all_h_prev[b])
        Z_next[b] = pca.transform(all_h_next[b])
        
    print("[*] Corpus generation complete.")
    return Z_prev, Z_next, all_x_in_scalar, all_delta_scalar

if __name__ == "__main__":
    # We'll run B=100 sequences to test the Invariant Hunter
    Z_prev, Z_next, X_in, Delta = extract_mass_trajectories(B=100, T=256)
    np.savez_compressed("mass_corpus.npz", z_prev=Z_prev, z_next=Z_next, x_in=X_in, delta=Delta)
    print("[*] Saved corpus to mass_corpus.npz")

import torch
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

temp_tensors = {}

def x_proj_hook(module, inputs, outputs):
    temp_tensors["x_in"] = inputs[0].detach().cpu().numpy()
    temp_tensors["x_proj_out"] = outputs.detach().cpu().numpy()

def extract_mamba_data(T=256, prompt_text=None):
    print("[*] Loading state-spaces/mamba-130m-hf...")
    model = AutoModelForCausalLM.from_pretrained('state-spaces/mamba-130m-hf')
    tokenizer = AutoTokenizer.from_pretrained('state-spaces/mamba-130m-hf')
    
    target_layer = 0
    mixer = model.backbone.layers[target_layer].mixer
    mixer.x_proj.register_forward_hook(x_proj_hook)
    
    if prompt_text is not None:
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
        print(f"[*] Initialized with prompt: '{prompt_text}' (Length: {input_ids.shape[1]})")
        # Ensure we have a valid 1D sequence to feed autoregressively.
        # Actually, if we just feed the last token of the prompt to start the generation:
        # We need to process the prompt first to get the cache state.
        if input_ids.shape[1] > 1:
            out = model(input_ids[:, :-1], use_cache=True)
            cache = out.cache_params
            input_ids = input_ids[:, -1:]
        else:
            cache = None
    else:
        input_ids = torch.randint(0, 50279, (1, 1))
        cache = None
    
    latent_data = {"h_prev": [], "h_next": [], "x_in": [], "delta": []}
    
    print(f"[*] Extracting T={T} recurrent transitions...")
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
    x_in_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["x_in"]], axis=0)
    delta_flat = np.concatenate([x.reshape(1, -1) for x in latent_data["delta"]], axis=0)
    
    print("[*] Bottlenecking to 8 Principal Components...")
    pca = PCA(n_components=8)
    pca.fit(h_prev_flat)
    
    z_prev = pca.transform(h_prev_flat)
    z_next = pca.transform(h_next_flat)
    x_in_scalar = np.mean(x_in_flat, axis=1)
    delta_scalar = np.mean(delta_flat, axis=1)
    
    N = z_prev.shape[0]
    X_matrix = np.zeros((N, 10), dtype=np.float64)
    X_matrix[:, :8] = z_prev
    X_matrix[:, 8] = x_in_scalar
    X_matrix[:, 9] = delta_scalar
    
    Y_matrix = z_next - z_prev
    
    print(f"[*] Formulated Ingestion Matrix X: {X_matrix.shape}, Target Y (all channels): {Y_matrix.shape}")
    return X_matrix, Y_matrix

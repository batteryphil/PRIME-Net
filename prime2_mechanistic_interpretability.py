import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
from prime_core import run_prime_engine
import gc

def extract_llm_thoughts():
    print("[*] Loading Neural Weights into VRAM...")
    
    # Try Llama-3 8B first, fallback to Qwen 0.5B if VRAM is insufficient
    try:
        model_id = "NousResearch/Meta-Llama-3-8B"
        print(f"[*] Attempting to load {model_id} (Requires ~15GB VRAM)")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        )
        layer_in, layer_out = 15, 16
    except Exception as e:
        print(f"[!] Llama-3 Load Failed (likely VRAM). Falling back to Qwen-0.5B. Error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        model_id = "Qwen/Qwen2.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16
        )
        layer_in, layer_out = 10, 11
        
    print(f"[*] Successfully hooked into {model_id}")
    
    # Complex Logical Stimulus to generate a rich latent thought space
    prompt = """
    Solve the following complex logic puzzle step-by-step:
    Three friends—Alice, Bob, and Charlie—each have a different favorite color (Red, Blue, Green) 
    and a different pet (Dog, Cat, Bird). 
    1. Alice does not like Red and does not own a Dog.
    2. The person who likes Blue owns a Cat.
    3. Bob owns a Bird.
    Therefore, mathematically deduce the color and pet of each person using predicate logic.
    """
    
    # We repeat it to ensure we get a large enough sequence length (>100 tokens)
    prompt = prompt * 3 
    
    print(f"[*] Injecting Stimulus ({len(tokenizer(prompt)['input_ids'])} tokens)...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("[*] Extracting Latent Thought Vectors...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    
    print(f"[*] Intercepting Neural Signals (Layer {layer_in} -> Layer {layer_out})")
    
    # Shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim)
    X_raw = hidden_states[layer_in][0].to(torch.float32).cpu().numpy()
    Y_raw = hidden_states[layer_out][0].to(torch.float32).cpu().numpy()
    
    print(f"[*] Raw Latent Dimension: {X_raw.shape[1]}")
    print("[*] Applying PCA Dimensionality Reduction to 5D Cognitive Manifold...")
    
    pca = PCA(n_components=5)
    X_features = pca.fit_transform(X_raw)
    
    # Target: The primary principal component of the thought transition
    Y_targets = pca.fit_transform(Y_raw)[:, 0]
    
    # Free VRAM just in case
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return X_features, Y_targets

def run_mechanistic_test():
    print("==========================================")
    print("PRIME-Net: MECHANISTIC INTERPRETABILITY (LLM MIND READER)")
    print("==========================================")
    
    X, y = extract_llm_thoughts()
    
    # Normalize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std
    
    print("[*] Data Normalized.")
    print(f"[*] X Shape: {X_norm.shape} (Layer In: 5D PCA Manifold)")
    print(f"[*] y Shape: {y_norm.shape} (Layer Out: Primary PC Transition)")
    
    print("\n[*] Launching PRIME-Net to discover the LLM's internal cognitive equation...")
    
    engine_config = {
        'pop_size': 1024,
        'seq_len': 63,
        'macro_seq_len': 15,
        'max_generations': 5000,
        'timeout_sec': 120.0 # 2 minutes
    }
    
    res = run_prime_engine(X_norm, y_norm, X_norm, y_norm, **engine_config)
    
    print("\n==========================================")
    print("TEST COMPLETE")
    print(f"Discovered Cognitive Invariant (Normalized): {res['best_sympy']}")
    print("==========================================")

if __name__ == "__main__":
    run_mechanistic_test()

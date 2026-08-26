import torch
import numpy as np
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_SIZE = 20

def decode_token(tok):
    mapping = {
        0:'X[t-1]', 1:'X[t-2]', 2:'X[t-3]', 3:'X[t-4]', 4:'X[t-5]', 5:'X[t-6]',
        6:'-1.0', 7:'0.1', 8:'0.5', 9:'1.0',
        10:'+', 11:'-', 12:'*', 13:'/', 14:'sin', 15:'avg',
        16:'exp', 17:'log', 18:'abs', 19:'NOP'
    }
    return mapping.get(tok, "?")

def rpn_to_str(seq):
    stack = []
    for t in seq:
        if t == 19: continue
        if t <= 9:
            stack.append(decode_token(t))
        elif t in [14, 16, 17, 18]:
            if len(stack) < 1: return "INVALID"
            a = stack.pop()
            op = decode_token(t)
            stack.append(f"{op}({a})")
        elif t in [10,11,12,13,15]:
            if len(stack) < 2: return "INVALID"
            b = stack.pop()
            a = stack.pop()
            op = decode_token(t)
            if t == 15:
                stack.append(f"avg({a}, {b})")
            else:
                stack.append(f"({a} {op} {b})")
    if len(stack) != 1: return "INVALID"
    return stack[0]

def evaluate_rpn(seq, X):
    stack = []
    X0, X1 = X[:,0], X[:,1]
    for t in seq:
        if t == 19: continue
        elif t == 0: stack.append(X0)
        elif t == 1: stack.append(X1)
        elif t >= 2 and t <= 5: stack.append(torch.zeros_like(X0))
        elif t == 6: stack.append(torch.full_like(X0, -1.0))
        elif t == 7: stack.append(torch.full_like(X0, 0.1))
        elif t == 8: stack.append(torch.full_like(X0, 0.5))
        elif t == 9: stack.append(torch.full_like(X0, 1.0))
        elif t == 14:
            if len(stack) < 1: return None
            stack.append(torch.sin(stack.pop()))
        elif t == 16:
            if len(stack) < 1: return None
            a = torch.clamp(stack.pop(), -10.0, 10.0)
            stack.append(torch.exp(a))
        elif t == 17:
            if len(stack) < 1: return None
            a = stack.pop()
            stack.append(torch.log(torch.abs(a) + 1e-5))
        elif t == 18:
            if len(stack) < 1: return None
            stack.append(torch.abs(stack.pop()))
        elif t in [10,11,12,13,15]:
            if len(stack) < 2: return None
            b = stack.pop()
            a = stack.pop()
            if t == 10: stack.append(a + b)
            elif t == 11: stack.append(a - b)
            elif t == 12: stack.append(a * b)
            elif t == 13:
                mask = (torch.abs(b) < 1e-5)
                b_safe = b.clone()
                b_safe[mask] = 1.0
                res = a / b_safe
                res[mask] = 1.0
                stack.append(res)
            elif t == 15:
                stack.append((a + b) / 2.0)
    if len(stack) != 1: return None
    return stack[0]

def generate_synthetic_data(domain=3.14, n_samples=2000, dataset_type="nonlinear"):
    X = torch.empty(n_samples, 2, dtype=torch.float32, device=device).uniform_(-domain, domain)
    if dataset_type == "nonlinear":
        Y = torch.sin(X[:,0]) + X[:,1]
    else: # multiplicative
        X = torch.empty(n_samples, 2, dtype=torch.float32, device=device).uniform_(1.0, 10.0)
        Y = X[:,0] * X[:,1]
        
    noise = torch.randn_like(Y) * 0.01
    Y += noise
    return X, Y

def run_prime(X_train, Y_train, ablation_mode="base", seed=0):
    torch.manual_seed(seed)
    pop_size = 512
    seq_len = 15
    generations = 2000
    
    master_idx = torch.randint(0, VOCAB_SIZE, (seq_len,), dtype=torch.int32, device=device)
    vote_buffer = torch.zeros(seq_len, dtype=torch.float32, device=device)
    step_sizes = torch.ones(seq_len, dtype=torch.int32, device=device)
    last_signs = torch.zeros(seq_len, dtype=torch.int32, device=device)
    
    mut_scale = 3
    target_flip_rate = 0.05
    threshold = 10.0 if ablation_mode != "no_vote" else 0.0
    lambda_penalty = 0.005 if ablation_mode != "no_complexity" else 0.0
    
    best_train_mse = float('inf')
    best_eq_overall = ""
    
    for gen in range(generations):
        if ablation_mode == "random_search":
            pop_idx = torch.randint(0, VOCAB_SIZE, (pop_size, seq_len), dtype=torch.int32, device=device)
            # Make sure NOP is still zero
            pop_idx[:, 0] = 0
        else:
            p = torch.randint(-mut_scale, mut_scale+1, (pop_size, seq_len), dtype=torch.int32, device=device)
            mask = (torch.rand(p.shape, device=device) < 0.20).int()
            p *= mask
            p[0] = 0
            pop_idx = torch.clamp(master_idx.unsqueeze(0) + p, 0, VOCAB_SIZE - 1)
            
        fits = torch.full((pop_size,), -1000000.0, dtype=torch.float32, device=device)
        pop_idx_cpu = pop_idx.cpu().tolist()
        
        for i in range(pop_size):
            eq_str = rpn_to_str(pop_idx_cpu[i])
            if eq_str == "INVALID":
                fits[i] = -1000000.0
                continue
                
            penalty = len(eq_str) * lambda_penalty
            
            y_pred = evaluate_rpn(pop_idx_cpu[i], X_train)
            if y_pred is not None:
                mse = torch.mean((y_pred - Y_train)**2)
                if torch.isnan(mse) or mse > 1000000:
                    fits[i] = -1000000.0
                else:
                    fits[i] = -mse.item() - penalty
                    
        if ablation_mode == "random_search":
            max_fit = fits.max().item()
            best_idx = fits.argmax().item()
            current_mse = -max_fit
            if current_mse < best_train_mse and current_mse > 0.0:
                best_train_mse = current_mse
                best_eq_overall = rpn_to_str(pop_idx[best_idx].cpu().tolist())
            continue
                    
        std_fit = fits.std().item()
        if std_fit < 1e-5:
            if ablation_mode == "no_random_walk":
                advs = torch.zeros_like(fits)
            else:
                advs = torch.randn_like(fits)
        else:
            advs = (fits - fits.mean()) / (fits.std() + 1e-8)
            
        vote_buffer += torch.sum((p.float() / float(mut_scale)) * advs.unsqueeze(1), dim=0)
        
        gamma = 0.9
        if std_fit < 1.0:
            gamma = 0.7
        elif std_fit > 100.0:
            gamma = 0.95
        gamma = max(0.70, min(0.95, gamma))
        
        if ablation_mode == "no_rprop":
            gamma = 0.0
            
        current_signs = torch.sign(vote_buffer).int()
        same_sign = (current_signs == last_signs) & (current_signs != 0)
        diff_sign = (current_signs != last_signs) & (current_signs != 0)
        
        if ablation_mode != "no_rprop":
            step_sizes[same_sign] = torch.clamp(step_sizes[same_sign] * 2, max=8)
            step_sizes[diff_sign] = torch.clamp(step_sizes[diff_sign] // 2, min=1)
        else:
            step_sizes = torch.ones(seq_len, dtype=torch.int32, device=device)
            
        last_signs[current_signs != 0] = current_signs[current_signs != 0]
        
        flips = (torch.abs(vote_buffer) > threshold).int()
        master_idx.data.add_(flips * current_signs * step_sizes).clamp_(0, VOCAB_SIZE - 1)
        
        vote_buffer[flips > 0] = 0
        vote_buffer.mul_(gamma)
        
        if ablation_mode != "no_vote":
            rate = flips.sum().item() / seq_len
            threshold = max(5.0, threshold * (1.0 + (rate - target_flip_rate)))
            
        max_fit = fits.max().item()
        best_idx = fits.argmax().item()
        current_mse = -max_fit
        
        if current_mse < best_train_mse and current_mse > 0.0:
            best_train_mse = current_mse
            best_eq_overall = rpn_to_str(pop_idx[best_idx].cpu().tolist())
            
    return best_eq_overall, best_train_mse

if __name__ == "__main__":
    modes = ["base", "no_rprop", "no_vote", "no_random_walk", "no_complexity", "random_search"]
    datasets = ["multiplicative", "nonlinear"]
    seeds = 5
    
    with open("ablation_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "ablation_mode", "seed", "final_equation", "mse"])
        
        for dataset in datasets:
            X, Y = generate_synthetic_data(dataset_type=dataset)
            for mode in modes:
                for seed in range(seeds):
                    eq, mse = run_prime(X, Y, mode, seed)
                    print(f"{dataset} | {mode:15s} | Seed {seed} | MSE: {mse:8.4f} | Eq: {eq}")
                    writer.writerow([dataset, mode, seed, eq, mse])

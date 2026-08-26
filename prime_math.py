import torch
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokens:
# 0: X
# 1-5: Constants 1.0, 2.0, 3.0, 4.0, 5.0
# 6: +
# 7: -
# 8: *
# 9: / (protected)
# 10: sin

VOCAB_SIZE = 11

def decode_token(tok):
    mapping = {0:'X', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 
               6:'+', 7:'-', 8:'*', 9:'/', 10:'sin'}
    return mapping.get(tok, "?")

def rpn_to_str(seq):
    stack = []
    for t in seq:
        if t <= 5:
            stack.append(decode_token(t))
        elif t == 10: # sin
            if len(stack) < 1: return "INVALID"
            a = stack.pop()
            stack.append(f"sin({a})")
        elif t in [6,7,8,9]:
            if len(stack) < 2: return "INVALID"
            b = stack.pop()
            a = stack.pop()
            op = decode_token(t)
            stack.append(f"({a} {op} {b})")
    if len(stack) != 1:
        return "INVALID"
    return stack[0]

def evaluate_rpn(seq, X):
    stack = []
    for t in seq:
        if t == 0:
            stack.append(X)
        elif 1 <= t <= 5:
            stack.append(torch.full_like(X, float(t)))
        elif t == 10:
            if len(stack) < 1: return None
            a = stack.pop()
            stack.append(torch.sin(a))
        elif t in [6,7,8,9]:
            if len(stack) < 2: return None
            b = stack.pop()
            a = stack.pop()
            if t == 6:
                stack.append(a + b)
            elif t == 7:
                stack.append(a - b)
            elif t == 8:
                stack.append(a * b)
            elif t == 9:
                mask = (torch.abs(b) < 1e-5)
                b_safe = b.clone()
                b_safe[mask] = 1.0
                res = a / b_safe
                res[mask] = 1.0
                stack.append(res)
    if len(stack) != 1:
        return None
    return stack[0]

def main():
    torch.manual_seed(42)
    pop_size = 512
    seq_len = 15
    generations = 5000
    
    # Target Equation: y = 2 * sin(x) + x
    X_data = torch.linspace(-10, 10, 100, device=device)
    Y_target = 2 * torch.sin(X_data) + X_data
    
    master_idx = torch.randint(0, VOCAB_SIZE, (seq_len,), dtype=torch.int32, device=device)
    
    vote_buffer = torch.zeros(seq_len, dtype=torch.float32, device=device)
    step_sizes = torch.ones(seq_len, dtype=torch.int32, device=device)
    last_signs = torch.zeros(seq_len, dtype=torch.int32, device=device)
    
    mut_scale = 3
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME Symbolic Regression...")
    
    best_score_overall = -float('inf')
    best_eq_overall = ""
    
    for gen in range(generations):
        # Perturbations
        p = torch.randint(-mut_scale, mut_scale+1, (pop_size, seq_len), dtype=torch.int32, device=device)
        mask = (torch.rand(p.shape, device=device) < 0.20).int()
        p *= mask
        p[0] = 0 # Master
        
        pop_idx = torch.clamp(master_idx.unsqueeze(0) + p, 0, VOCAB_SIZE - 1)
        
        fits = torch.full((pop_size,), -1000.0, dtype=torch.float32, device=device)
        
        pop_idx_cpu = pop_idx.cpu().tolist()
        for i in range(pop_size):
            y_pred = evaluate_rpn(pop_idx_cpu[i], X_data)
            if y_pred is not None:
                mse = torch.mean((y_pred - Y_target)**2)
                if torch.isnan(mse) or mse > 1000:
                    fits[i] = -1000.0
                else:
                    fits[i] = -mse.item()
                    
        std_fit = fits.std().item()
        if std_fit < 1e-5:
            # Zero-Gradient Void (e.g. all ghosts generated INVALID strings)
            # Inject random noise so the swarm randomly walks until someone finds a valid equation
            advs = torch.randn_like(fits)
        else:
            advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        vote_buffer += torch.sum((p.float() / float(mut_scale)) * advs.unsqueeze(1), dim=0)
        
        gamma = 0.9
        std_fit = fits.std().item()
        if std_fit < 1.0:
            gamma = 0.7
        elif std_fit > 100.0:
            gamma = 0.95
        gamma = max(0.70, min(0.95, gamma))
        
        current_signs = torch.sign(vote_buffer).int()
        same_sign = (current_signs == last_signs) & (current_signs != 0)
        diff_sign = (current_signs != last_signs) & (current_signs != 0)
        
        step_sizes[same_sign] = torch.clamp(step_sizes[same_sign] * 2, max=8)
        step_sizes[diff_sign] = torch.clamp(step_sizes[diff_sign] // 2, min=1)
        
        last_signs[current_signs != 0] = current_signs[current_signs != 0]
        
        flips = (torch.abs(vote_buffer) > threshold).int()
        master_idx.data.add_(flips * current_signs * step_sizes).clamp_(0, VOCAB_SIZE - 1)
        
        vote_buffer[flips > 0] = 0
        vote_buffer.mul_(gamma)
        
        rate = flips.sum().item() / seq_len
        threshold = max(5.0, threshold * (1.0 + (rate - target_flip_rate)))
        
        max_fit = fits.max().item()
        mean_fit = fits.mean().item()
        
        if max_fit > best_score_overall:
            best_score_overall = max_fit
            best_eq_overall = rpn_to_str(master_idx.cpu().tolist())
            
        if gen % 10 == 0:
            eq_str = rpn_to_str(master_idx.cpu().tolist())
            print(f"Gen {gen:4d} | Mean Fit: {mean_fit:8.2f} | Max Fit: {max_fit:8.2f} | Flips: {flips.sum().item():2d} | Thresh: {threshold:5.1f} | Eq: {eq_str}")
            
        if max_fit >= -1e-4:
            print("Perfect Equation Discovered!")
            print(f"Eq: {rpn_to_str(master_idx.cpu().tolist())}")
            break

if __name__ == "__main__":
    main()

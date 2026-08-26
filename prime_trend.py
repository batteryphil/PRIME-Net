import torch
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = 14

def decode_token(tok):
    mapping = {0:'X[t-1]', 1:'X[t-2]', 2:'X[t-3]', 
               3:'0.1', 4:'0.5', 5:'1.0', 6:'2.0', 7:'5.0',
               8:'+', 9:'-', 10:'*', 11:'/', 12:'sin', 13:'avg'}
    return mapping.get(tok, "?")

def rpn_to_str(seq):
    stack = []
    for t in seq:
        if t <= 7:
            stack.append(decode_token(t))
        elif t == 12: # sin
            if len(stack) < 1: return "INVALID"
            a = stack.pop()
            stack.append(f"sin({a})")
        elif t in [8,9,10,11,13]:
            if len(stack) < 2: return "INVALID"
            b = stack.pop()
            a = stack.pop()
            op = decode_token(t)
            if t == 13:
                stack.append(f"avg({a}, {b})")
            else:
                stack.append(f"({a} {op} {b})")
    if len(stack) != 1:
        return "INVALID"
    return stack[0]

def evaluate_rpn(seq, X0, X1, X2):
    stack = []
    for t in seq:
        if t == 0: stack.append(X0)
        elif t == 1: stack.append(X1)
        elif t == 2: stack.append(X2)
        elif t == 3: stack.append(torch.full_like(X0, 0.1))
        elif t == 4: stack.append(torch.full_like(X0, 0.5))
        elif t == 5: stack.append(torch.full_like(X0, 1.0))
        elif t == 6: stack.append(torch.full_like(X0, 2.0))
        elif t == 7: stack.append(torch.full_like(X0, 5.0))
        elif t == 12: # sin
            if len(stack) < 1: return None
            stack.append(torch.sin(stack.pop()))
        elif t in [8,9,10,11,13]:
            if len(stack) < 2: return None
            b = stack.pop()
            a = stack.pop()
            if t == 8: stack.append(a + b)
            elif t == 9: stack.append(a - b)
            elif t == 10: stack.append(a * b)
            elif t == 11:
                mask = (torch.abs(b) < 1e-5)
                b_safe = b.clone()
                b_safe[mask] = 1.0
                res = a / b_safe
                res[mask] = 1.0
                stack.append(res)
            elif t == 13:
                stack.append((a + b) / 2.0)
    if len(stack) != 1: return None
    return stack[0]

def main():
    torch.manual_seed(42)
    pop_size = 512
    seq_len = 15
    generations = 5000
    
    # Generate Synthetic Time-Series Data (Linear Trend + Cyclic + Noise)
    T = 200
    timeline = torch.linspace(0, 20, T, device=device)
    raw_price = 100 + 5 * timeline + 10 * torch.sin(timeline) + torch.randn(T, device=device) * 2.0
    
    X0_list, X1_list, X2_list, Y_list = [], [], [], []
    for t in range(3, T):
        X0_list.append(raw_price[t-1])
        X1_list.append(raw_price[t-2])
        X2_list.append(raw_price[t-3])
        Y_list.append(raw_price[t])
        
    X0 = torch.stack(X0_list)
    X1 = torch.stack(X1_list)
    X2 = torch.stack(X2_list)
    Y = torch.stack(Y_list)
    
    # Train / Val Split (80% / 20%)
    split_idx = int(len(Y) * 0.8)
    X0_tr, X1_tr, X2_tr, Y_tr = X0[:split_idx], X1[:split_idx], X2[:split_idx], Y[:split_idx]
    X0_val, X1_val, X2_val, Y_val = X0[split_idx:], X1[split_idx:], X2[split_idx:], Y[split_idx:]
    
    master_idx = torch.randint(0, VOCAB_SIZE, (seq_len,), dtype=torch.int32, device=device)
    vote_buffer = torch.zeros(seq_len, dtype=torch.float32, device=device)
    step_sizes = torch.ones(seq_len, dtype=torch.int32, device=device)
    last_signs = torch.zeros(seq_len, dtype=torch.int32, device=device)
    
    mut_scale = 3
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME Trend Prediction Engine...")
    
    best_train_mse = float('inf')
    best_val_mse = float('inf')
    best_eq_overall = ""
    
    for gen in range(generations):
        # Perturbations
        p = torch.randint(-mut_scale, mut_scale+1, (pop_size, seq_len), dtype=torch.int32, device=device)
        mask = (torch.rand(p.shape, device=device) < 0.20).int()
        p *= mask
        p[0] = 0 # Master
        
        pop_idx = torch.clamp(master_idx.unsqueeze(0) + p, 0, VOCAB_SIZE - 1)
        
        fits = torch.full((pop_size,), -1000000.0, dtype=torch.float32, device=device)
        
        pop_idx_cpu = pop_idx.cpu().tolist()
        for i in range(pop_size):
            y_pred = evaluate_rpn(pop_idx_cpu[i], X0_tr, X1_tr, X2_tr)
            if y_pred is not None:
                mse = torch.mean((y_pred - Y_tr)**2)
                if torch.isnan(mse) or mse > 1000000:
                    fits[i] = -1000000.0
                else:
                    fits[i] = -mse.item()
                    
        std_fit = fits.std().item()
        if std_fit < 1e-5:
            # Zero-Gradient Void Injection
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
        
        max_fit = fits.max().item() # Max fit is negative MSE
        current_mse = -max_fit
        
        if current_mse < best_train_mse and current_mse > 0.0:
            best_train_mse = current_mse
            # Calculate validation MSE for this master
            val_pred = evaluate_rpn(master_idx.cpu().tolist(), X0_val, X1_val, X2_val)
            if val_pred is not None:
                val_mse = torch.mean((val_pred - Y_val)**2).item()
                best_val_mse = val_mse
            best_eq_overall = rpn_to_str(master_idx.cpu().tolist())
            
        if gen % 50 == 0:
            eq_str = rpn_to_str(master_idx.cpu().tolist())
            print(f"Gen {gen:4d} | Train MSE: {current_mse:9.2f} | Best Val MSE: {best_val_mse:9.2f} | Flips: {flips.sum().item():2d} | Thresh: {threshold:5.1f} | Eq: {eq_str}")

if __name__ == "__main__":
    main()

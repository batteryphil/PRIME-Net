import urllib.request
import torch
import numpy as np

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
        if t == 19:
            continue
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
    if len(stack) != 1:
        return "INVALID"
    return stack[0]

def evaluate_rpn(seq, X0, X1, X2, X3, X4, X5):
    stack = []
    for t in seq:
        if t == 19: continue
        elif t == 0: stack.append(X0)
        elif t == 1: stack.append(X1)
        elif t == 2: stack.append(X2)
        elif t == 3: stack.append(X3)
        elif t == 4: stack.append(X4)
        elif t == 5: stack.append(X5)
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

def prepare_data():
    print("Downloading NOAA Nino 3.4 SST dataset...")
    url = "https://psl.noaa.gov/data/correlation/nina34.data"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        lines = response.read().decode('utf-8').split('\n')
        
    start_year, end_year = map(int, lines[0].strip().split())
    
    data = []
    for i in range(1, len(lines)):
        parts = lines[i].strip().split()
        if len(parts) == 13:
            year = int(parts[0])
            temps = [float(x) for x in parts[1:]]
            for t in temps:
                if t > -50: # -99.99 is missing
                    data.append((year, t))
                    
    data_deltas = []
    for i in range(1, len(data)):
        data_deltas.append((data[i][0], data[i][1] - data[i-1][1]))
        
    print(f"Calculated {len(data_deltas)} monthly SST deltas (rate of change).")
    
    X0, X1, X2, X3, X4, X5, Y, years = [], [], [], [], [], [], [], []
    for t in range(6, len(data_deltas)):
        X5.append(data_deltas[t-6][1])
        X4.append(data_deltas[t-5][1])
        X3.append(data_deltas[t-4][1])
        X2.append(data_deltas[t-3][1])
        X1.append(data_deltas[t-2][1])
        X0.append(data_deltas[t-1][1])
        Y.append(data_deltas[t][1])
        years.append(data_deltas[t][0])
        
    # Split
    split_idx = -1
    for i, y in enumerate(years):
        if y >= 2019:
            split_idx = i
            break
            
    print(f"Training on years {years[0]} to {years[split_idx-1]} ({split_idx} months)")
    print(f"Testing on years {years[split_idx]} to {years[-1]} ({len(years) - split_idx} months)")
    
    X0_tr = torch.tensor(X0[:split_idx], dtype=torch.float32, device=device)
    X1_tr = torch.tensor(X1[:split_idx], dtype=torch.float32, device=device)
    X2_tr = torch.tensor(X2[:split_idx], dtype=torch.float32, device=device)
    X3_tr = torch.tensor(X3[:split_idx], dtype=torch.float32, device=device)
    X4_tr = torch.tensor(X4[:split_idx], dtype=torch.float32, device=device)
    X5_tr = torch.tensor(X5[:split_idx], dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y[:split_idx], dtype=torch.float32, device=device)
    
    X0_te = torch.tensor(X0[split_idx:], dtype=torch.float32, device=device)
    X1_te = torch.tensor(X1[split_idx:], dtype=torch.float32, device=device)
    X2_te = torch.tensor(X2[split_idx:], dtype=torch.float32, device=device)
    X3_te = torch.tensor(X3[split_idx:], dtype=torch.float32, device=device)
    X4_te = torch.tensor(X4[split_idx:], dtype=torch.float32, device=device)
    X5_te = torch.tensor(X5[split_idx:], dtype=torch.float32, device=device)
    Y_te = torch.tensor(Y[split_idx:], dtype=torch.float32, device=device)
    
    return X0_tr, X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, Y_tr, X0_te, X1_te, X2_te, X3_te, X4_te, X5_te, Y_te

def main():
    torch.manual_seed(42)
    pop_size = 512
    seq_len = 15
    generations = 5000
    
    X0_tr, X1_tr, X2_tr, X3_tr, X4_tr, X5_tr, Y_tr, X0_te, X1_te, X2_te, X3_te, X4_te, X5_te, Y_te = prepare_data()
    
    master_idx = torch.randint(0, VOCAB_SIZE, (seq_len,), dtype=torch.int32, device=device)
    vote_buffer = torch.zeros(seq_len, dtype=torch.float32, device=device)
    step_sizes = torch.ones(seq_len, dtype=torch.int32, device=device)
    last_signs = torch.zeros(seq_len, dtype=torch.int32, device=device)
    
    mut_scale = 3
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME El Nino Predictor...")
    
    best_train_mse = float('inf')
    best_eq_overall = ""
    best_seq_overall = []
    
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
            eq_str = rpn_to_str(pop_idx_cpu[i])
            if eq_str == "INVALID":
                fits[i] = -1000000.0
                continue
                
            penalty = len(eq_str) * 0.005
            
            y_pred = evaluate_rpn(pop_idx_cpu[i], X0_tr, X1_tr, X2_tr, X3_tr, X4_tr, X5_tr)
            if y_pred is not None:
                mse = torch.mean((y_pred - Y_tr)**2)
                if torch.isnan(mse) or mse > 1000000:
                    fits[i] = -1000000.0
                else:
                    fits[i] = -mse.item() - penalty
                    
        std_fit = fits.std().item()
        if std_fit < 1e-5:
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
        
        max_fit = fits.max().item()
        current_mse = -max_fit
        
        if current_mse < best_train_mse and current_mse > 0.0:
            best_train_mse = current_mse
            best_seq_overall = master_idx.cpu().tolist()
            best_eq_overall = rpn_to_str(best_seq_overall)
            
        if gen % 500 == 0:
            print(f"Gen {gen:4d} | Train MSE: {current_mse:9.2f} | Eq: {rpn_to_str(master_idx.cpu().tolist())}")

    print("\n========== EVALUATION ON UNSEEN ENSO CYCLE (2019-2023) ==========")
    print(f"Final Equation Evolved: {best_eq_overall}")
    print(f"Training MSE (1948-2018): {best_train_mse:.4f}")
    
    # Calculate Prediction on unseen Test set
    y_pred_holdout = evaluate_rpn(best_seq_overall, X0_te, X1_te, X2_te, X3_te, X4_te, X5_te)
    
    if y_pred_holdout is not None:
        val_mse = torch.mean((y_pred_holdout - Y_te)**2).item()
        baseline_mse = torch.mean((0.0 - Y_te)**2).item() # Baseline: Predicting 0 delta (no change)
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Naive Baseline MSE: {baseline_mse:.4f} (Predicting a flat 0.0 delta)")
        if val_mse < baseline_mse:
            print("SUCCESS: The model successfully deciphered physical dynamics and beat the baseline!")
        else:
            print("FAILURE: The model failed to beat a naive 'no change' baseline.")
    else:
        print("Final equation is invalid and cannot be evaluated.")

if __name__ == "__main__":
    main()

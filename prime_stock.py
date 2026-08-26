import yfinance as yf
import pandas as pd
import torch
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOCAB_SIZE = 16

def decode_token(tok):
    mapping = {
        0:'X[t-1]', 1:'X[t-2]', 2:'X[t-3]', 3:'X[t-4]', 4:'X[t-5]',
        5:'-1.0', 6:'0.1', 7:'0.5', 8:'1.0', 9:'2.0',
        10:'+', 11:'-', 12:'*', 13:'/', 14:'sin', 15:'avg'
    }
    return mapping.get(tok, "?")

def rpn_to_str(seq):
    stack = []
    for t in seq:
        if t <= 9:
            stack.append(decode_token(t))
        elif t == 14: # sin
            if len(stack) < 1: return "INVALID"
            a = stack.pop()
            stack.append(f"sin({a})")
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

def evaluate_rpn(seq, X0, X1, X2, X3, X4):
    stack = []
    for t in seq:
        if t == 0: stack.append(X0)
        elif t == 1: stack.append(X1)
        elif t == 2: stack.append(X2)
        elif t == 3: stack.append(X3)
        elif t == 4: stack.append(X4)
        elif t == 5: stack.append(torch.full_like(X0, -1.0))
        elif t == 6: stack.append(torch.full_like(X0, 0.1))
        elif t == 7: stack.append(torch.full_like(X0, 0.5))
        elif t == 8: stack.append(torch.full_like(X0, 1.0))
        elif t == 9: stack.append(torch.full_like(X0, 2.0))
        elif t == 14:
            if len(stack) < 1: return None
            stack.append(torch.sin(stack.pop()))
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
    print("Downloading historical SPY data from Yahoo Finance...")
    spy = yf.download('SPY', start='2020-01-01', end='2023-12-05', progress=False)
    closes = spy['Close'].values.flatten()
    dates = spy.index
    
    # Daily returns %
    returns = []
    for i in range(1, len(closes)):
        returns.append( ((closes[i] - closes[i-1]) / closes[i-1]) * 100.0 )
    
    ret_dates = dates[1:]
    
    X0, X1, X2, X3, X4, Y, valid_dates = [], [], [], [], [], [], []
    for t in range(5, len(returns)):
        X4.append(returns[t-5])
        X3.append(returns[t-4])
        X2.append(returns[t-3])
        X1.append(returns[t-2])
        X0.append(returns[t-1])
        Y.append(returns[t])
        valid_dates.append(ret_dates[t])
        
    # Split index for end of Nov 2023
    split_idx = -1
    for i, d in enumerate(valid_dates):
        if d.year == 2023 and d.month == 11 and d.day >= 29:
            split_idx = i
            break
            
    if split_idx == -1:
        # Fallback
        for i, d in enumerate(valid_dates):
            if d.year == 2023 and d.month == 12:
                split_idx = i - 1
                break

    print(f"Training Data: 2020-01-01 through {valid_dates[split_idx].date()}")
    
    X0_tr = torch.tensor(X0[:split_idx+1], dtype=torch.float32, device=device)
    X1_tr = torch.tensor(X1[:split_idx+1], dtype=torch.float32, device=device)
    X2_tr = torch.tensor(X2[:split_idx+1], dtype=torch.float32, device=device)
    X3_tr = torch.tensor(X3[:split_idx+1], dtype=torch.float32, device=device)
    X4_tr = torch.tensor(X4[:split_idx+1], dtype=torch.float32, device=device)
    Y_tr = torch.tensor(Y[:split_idx+1], dtype=torch.float32, device=device)
    
    X0_te = torch.tensor([X0[split_idx+1]], dtype=torch.float32, device=device)
    X1_te = torch.tensor([X1[split_idx+1]], dtype=torch.float32, device=device)
    X2_te = torch.tensor([X2[split_idx+1]], dtype=torch.float32, device=device)
    X3_te = torch.tensor([X3[split_idx+1]], dtype=torch.float32, device=device)
    X4_te = torch.tensor([X4[split_idx+1]], dtype=torch.float32, device=device)
    Y_te = torch.tensor([Y[split_idx+1]], dtype=torch.float32, device=device)
    target_date = valid_dates[split_idx+1]
    
    print(f"Holdout Target Date: {target_date.date()}")
    
    return X0_tr, X1_tr, X2_tr, X3_tr, X4_tr, Y_tr, X0_te, X1_te, X2_te, X3_te, X4_te, Y_te, target_date

def main():
    torch.manual_seed(42)
    pop_size = 512
    seq_len = 15
    generations = 5000
    
    X0_tr, X1_tr, X2_tr, X3_tr, X4_tr, Y_tr, X0_te, X1_te, X2_te, X3_te, X4_te, Y_te, target_date = prepare_data()
    
    master_idx = torch.randint(0, VOCAB_SIZE, (seq_len,), dtype=torch.int32, device=device)
    vote_buffer = torch.zeros(seq_len, dtype=torch.float32, device=device)
    step_sizes = torch.ones(seq_len, dtype=torch.int32, device=device)
    last_signs = torch.zeros(seq_len, dtype=torch.int32, device=device)
    
    mut_scale = 3
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME Stock Predictor...")
    
    best_train_mse = float('inf')
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
            y_pred = evaluate_rpn(pop_idx_cpu[i], X0_tr, X1_tr, X2_tr, X3_tr, X4_tr)
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
            best_eq_overall = rpn_to_str(master_idx.cpu().tolist())
            
        if gen % 500 == 0:
            print(f"Gen {gen:4d} | Train MSE: {current_mse:9.2f} | Eq: {rpn_to_str(master_idx.cpu().tolist())}")

    print("\n========== EVALUATION ON UNSEEN FUTURE ==========")
    print(f"Final Equation Evolved: {best_eq_overall}")
    print(f"Target Date: {target_date.date()}")
    
    # Calculate Prediction
    best_seq = master_idx.cpu().tolist()
    y_pred_holdout = evaluate_rpn(best_seq, X0_te, X1_te, X2_te, X3_te, X4_te)
    
    if y_pred_holdout is not None:
        pred_val = y_pred_holdout.item()
        actual_val = Y_te.item()
        print(f"Predicted SPY Return: {pred_val:+.2f}%")
        print(f"Actual SPY Return:    {actual_val:+.2f}%")
        
        # Check directional accuracy
        if (pred_val > 0 and actual_val > 0) or (pred_val < 0 and actual_val < 0):
            print("Direction: CORRECT")
        else:
            print("Direction: INCORRECT")
    else:
        print("Final equation is invalid and cannot be evaluated.")

if __name__ == "__main__":
    main()

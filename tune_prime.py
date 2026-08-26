import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import itertools

# --- PRIME Components ---
def get_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > num: break
            if num % p == 0:
                is_prime = False
                break
        if is_prime: primes.append(num)
        num += 1
    return primes

def build_prime_lut(size=65536):
    primes = get_primes(size // 2)
    pos = [1.0 / p for p in primes]
    neg = [-1.0 / p for p in primes]
    lut = sorted(pos + neg)
    return torch.tensor(lut, dtype=torch.float32)

LUT = build_prime_lut(size=65536)
LUT_SIZE = len(LUT)

class PrimeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.idx = nn.Parameter(torch.randint(0, LUT_SIZE, (out_features, in_features), dtype=torch.int32), requires_grad=False)
        self.bias_idx = nn.Parameter(torch.randint(0, LUT_SIZE, (out_features,), dtype=torch.int32), requires_grad=False)
        self.vote_buffer = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_vote_buffer = torch.zeros((out_features,), dtype=torch.float32)

    def forward(self, x, w_p=None, b_p=None):
        if w_p is None:
            w = LUT[self.idx]
            b = LUT[self.bias_idx]
        else:
            w = LUT[torch.clamp(self.idx + w_p, 0, LUT_SIZE - 1)]
            b = LUT[torch.clamp(self.bias_idx + b_p, 0, LUT_SIZE - 1)]
        return nn.functional.linear(x, w, b)

class PrimeAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = PrimeLinear(4, 2)
    def forward(self, x, p=None):
        if p is None: return self.layer(x)
        return self.layer(x, p[0], p[1])

def run_episode(env, agent, p=None):
    obs, _ = env.reset()
    total_reward = 0
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = agent(obs_t, p)
            action = torch.argmax(q, dim=1).item()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated: break
    return total_reward

# --- Tuning Loop ---
def test_config(mut_scale, step_size, target_flip, pop_size=64, max_gens=40):
    env = gym.make('CartPole-v1')
    master = PrimeAgent()
    threshold = 10.0
    
    max_fit = 0
    solved_gen = max_gens
    success_count = 0
    
    for gen in range(max_gens):
        pop_p = [(torch.randint(-mut_scale, mut_scale+1, master.layer.idx.shape, dtype=torch.int32),
                  torch.randint(-mut_scale, mut_scale+1, master.layer.bias_idx.shape, dtype=torch.int32)) for _ in range(pop_size)]
        
        fits = np.array([run_episode(env, master, p) for p in pop_p])
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        for i, (w_p, b_p) in enumerate(pop_p):
            master.layer.vote_buffer += (w_p.float() / float(mut_scale)) * advs[i]
            master.layer.bias_vote_buffer += (b_p.float() / float(mut_scale)) * advs[i]
            
        total_flips, total_params = 0, 0
        for p_idx, v_buf in [(master.layer.idx, master.layer.vote_buffer), (master.layer.bias_idx, master.layer.bias_vote_buffer)]:
            flips = (torch.abs(v_buf) > threshold).int()
            signs = torch.sign(v_buf).int()
            p_idx.data.add_(flips * signs * step_size).clamp_(0, LUT_SIZE - 1)
            v_buf[flips > 0] = 0
            v_buf.mul_(0.9)
            total_flips += flips.sum().item()
            total_params += p_idx.numel()
            
        rate = total_flips / total_params
        threshold = max(5.0, threshold * (1.0 + (rate - target_flip) * 10.0))
        
        master_fit = run_episode(env, master)
        max_fit = max(max_fit, master_fit)
        
        if master_fit >= 500:
            success_count += 1
            if success_count >= 2:
                solved_gen = gen
                break
        else:
            success_count = 0
            
    env.close()
    return max_fit, solved_gen

def run_grid_search():
    mut_scales = [1000, 2000, 4000]
    step_sizes = [500, 1000, 2000, 4000]
    target_flips = [0.05, 0.1, 0.2]
    
    results = []
    
    configs = list(itertools.product(mut_scales, step_sizes, target_flips))
    print(f"Starting grid search over {len(configs)} configurations...")
    
    for i, (mut, step, flip) in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Testing Mut: {mut}, Step: {step}, Flip: {flip}...")
        # Run 2 short trials per config to rule out lucky seeds
        f1, g1 = test_config(mut, step, flip)
        f2, g2 = test_config(mut, step, flip)
        
        mean_fit = (f1 + f2) / 2.0
        mean_gen = (g1 + g2) / 2.0
        
        results.append({
            'Mutation Scale': mut,
            'Step Size': step,
            'Target Flip Rate': flip,
            'Mean Max Fitness': mean_fit,
            'Mean Gens to Solve': mean_gen
        })
        
    df = pd.DataFrame(results)
    df = df.sort_values(by=['Mean Max Fitness', 'Mean Gens to Solve'], ascending=[False, True])
    
    print("\\n--- TOP 5 CONFIGURATIONS ---")
    print(df.head(5).to_string(index=False))
    df.to_csv('tuning_results.csv', index=False)

if __name__ == '__main__':
    run_grid_search()

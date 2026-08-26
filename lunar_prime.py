import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

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

class LunarAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PrimeLinear(8, 32)
        self.layer2 = PrimeLinear(32, 4)
        
    def forward(self, x, p=None):
        if p is None:
            x = torch.relu(self.layer1(x))
            return self.layer2(x)
        else:
            w1_p, b1_p, w2_p, b2_p = p
            x = torch.relu(self.layer1(x, w1_p, b1_p))
            return self.layer2(x, w2_p, b2_p)

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

# --- Training Loop ---
def train():
    env = gym.make('LunarLander-v3')
    master = LunarAgent()
    
    pop_size = 64
    generations = 300
    
    # Golden Configuration for Multi-Layer
    mut_scale = 1000
    step_size = 1000
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME Genetic Evolution on LunarLander...")
    print(f"Network: 8 -> 32 -> 4 (Total Params: {8*32 + 32 + 32*4 + 4})")
    
    for gen in range(generations):
        t0 = time.time()
        
        pop_p = []
        for _ in range(pop_size):
            w1_p = torch.randint(-mut_scale, mut_scale+1, master.layer1.idx.shape, dtype=torch.int32)
            b1_p = torch.randint(-mut_scale, mut_scale+1, master.layer1.bias_idx.shape, dtype=torch.int32)
            w2_p = torch.randint(-mut_scale, mut_scale+1, master.layer2.idx.shape, dtype=torch.int32)
            b2_p = torch.randint(-mut_scale, mut_scale+1, master.layer2.bias_idx.shape, dtype=torch.int32)
            pop_p.append((w1_p, b1_p, w2_p, b2_p))
            
        fits = np.array([run_episode(env, master, p) for p in pop_p])
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        for i, (w1_p, b1_p, w2_p, b2_p) in enumerate(pop_p):
            master.layer1.vote_buffer += (w1_p.float() / float(mut_scale)) * advs[i]
            master.layer1.bias_vote_buffer += (b1_p.float() / float(mut_scale)) * advs[i]
            master.layer2.vote_buffer += (w2_p.float() / float(mut_scale)) * advs[i]
            master.layer2.bias_vote_buffer += (b2_p.float() / float(mut_scale)) * advs[i]
            
        total_flips, total_params = 0, 0
        layers = [master.layer1, master.layer2]
        
        for layer in layers:
            for p_idx, v_buf in [(layer.idx, layer.vote_buffer), (layer.bias_idx, layer.bias_vote_buffer)]:
                flips = (torch.abs(v_buf) > threshold).int()
                signs = torch.sign(v_buf).int()
                p_idx.data.add_(flips * signs * step_size).clamp_(0, LUT_SIZE - 1)
                v_buf[flips > 0] = 0
                v_buf.mul_(0.9) # Momentum decay
                total_flips += flips.sum().item()
                total_params += p_idx.numel()
                
        rate = total_flips / total_params
        # Reduce PI gain from 10.0 to 1.0 to prevent oscillation
        threshold = max(5.0, threshold * (1.0 + (rate - target_flip_rate)))
        
        if gen % 5 == 0:
            render_env = gym.make('LunarLander-v3', render_mode='human')
            master_fit = run_episode(render_env, master)
            render_env.close()
        else:
            master_fit = run_episode(env, master)
            
        print(f"Gen {gen:3d} | Mean Fit: {fits.mean():7.1f} | Master Fit: {master_fit:7.1f} | Flips: {total_flips:4d} ({rate*100:5.2f}%) | Thresh: {threshold:5.1f} | Time: {time.time()-t0:.2f}s")
        
        if master_fit >= 200:
            print(f"LunarLander SOLVED in {gen} generations!")
            break
            
    env.close()

if __name__ == '__main__':
    train()

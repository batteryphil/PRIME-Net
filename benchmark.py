import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

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

# --- Continuous ES Components ---

class ContinuousAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 2)
        # Initialize randomly similarly to LUT range [-0.5, 0.5]
        nn.init.uniform_(self.layer.weight, -0.5, 0.5)
        nn.init.uniform_(self.layer.bias, -0.5, 0.5)
    
    def forward(self, x, p=None):
        if p is None: return self.layer(x)
        w = self.layer.weight + p[0]
        b = self.layer.bias + p[1]
        return nn.functional.linear(x, w, b)

# --- Evaluation ---

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

# --- Training Loops ---

def train_prime(generations, pop_size):
    env = gym.make('CartPole-v1')
    master = PrimeAgent()
    threshold = 10.0
    history = []
    
    for gen in range(generations):
        pop_p = [(torch.randint(-2000, 2001, master.layer.idx.shape, dtype=torch.int32),
                  torch.randint(-2000, 2001, master.layer.bias_idx.shape, dtype=torch.int32)) for _ in range(pop_size)]
        
        fits = np.array([run_episode(env, master, p) for p in pop_p])
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        for i, (w_p, b_p) in enumerate(pop_p):
            master.layer.vote_buffer += (w_p.float() / 2000.0) * advs[i]
            master.layer.bias_vote_buffer += (b_p.float() / 2000.0) * advs[i]
            
        total_flips, total_params = 0, 0
        for p_idx, v_buf in [(master.layer.idx, master.layer.vote_buffer), (master.layer.bias_idx, master.layer.bias_vote_buffer)]:
            flips = (torch.abs(v_buf) > threshold).int()
            signs = torch.sign(v_buf).int()
            p_idx.data.add_(flips * signs * 4000).clamp_(0, LUT_SIZE - 1)
            v_buf[flips > 0] = 0
            v_buf.mul_(0.9)
            total_flips += flips.sum().item()
            total_params += p_idx.numel()
            
        rate = total_flips / total_params
        threshold = max(5.0, threshold * (1.0 + (rate - 0.2) * 10.0))
        master_fit = run_episode(env, master)
        history.append(master_fit)
        
    env.close()
    return history

def train_continuous(generations, pop_size):
    env = gym.make('CartPole-v1')
    master = ContinuousAgent()
    lr = 0.05
    sigma = 0.1
    history = []
    
    for gen in range(generations):
        pop_p = [(torch.randn_like(master.layer.weight) * sigma,
                  torch.randn_like(master.layer.bias) * sigma) for _ in range(pop_size)]
        
        fits = np.array([run_episode(env, master, p) for p in pop_p])
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        w_grad = torch.zeros_like(master.layer.weight)
        b_grad = torch.zeros_like(master.layer.bias)
        
        for i, (w_p, b_p) in enumerate(pop_p):
            w_grad += w_p * advs[i]
            b_grad += b_p * advs[i]
            
        w_grad /= (pop_size * sigma)
        b_grad /= (pop_size * sigma)
        
        with torch.no_grad():
            master.layer.weight += lr * w_grad
            master.layer.bias += lr * b_grad
            
        master_fit = run_episode(env, master)
        history.append(master_fit)
        
    env.close()
    return history

# --- Main Benchmark ---
def run_benchmark():
    trials = 5
    generations = 100
    pop_size = 64
    
    results = []
    
    print("Running PRIME ES...")
    for t in range(trials):
        print(f" Trial {t+1}/{trials}")
        torch.manual_seed(t)
        np.random.seed(t)
        h = train_prime(generations, pop_size)
        for g, fit in enumerate(h):
            results.append({'Algorithm': 'PRIME', 'Trial': t, 'Generation': g, 'Fitness': fit})
            
    print("Running Continuous ES...")
    for t in range(trials):
        print(f" Trial {t+1}/{trials}")
        torch.manual_seed(t)
        np.random.seed(t)
        h = train_continuous(generations, pop_size)
        for g, fit in enumerate(h):
            results.append({'Algorithm': 'Continuous ES', 'Trial': t, 'Generation': g, 'Fitness': fit})
            
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)
    
    # Plotting
    mean_df = df.groupby(['Algorithm', 'Generation'])['Fitness'].mean().reset_index()
    std_df = df.groupby(['Algorithm', 'Generation'])['Fitness'].std().reset_index()
    
    plt.figure(figsize=(10, 6))
    for algo in ['PRIME', 'Continuous ES']:
        m = mean_df[mean_df['Algorithm'] == algo]['Fitness'].values
        s = std_df[std_df['Algorithm'] == algo]['Fitness'].values
        plt.plot(m, label=f"{algo} (Mean)")
        plt.fill_between(range(generations), m - s, m + s, alpha=0.2)
        
    plt.title('PRIME Genetic Evolution vs Continuous ES (CartPole)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Max 500)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('benchmark_plot.png', dpi=300, bbox_inches='tight')
    print("Benchmark complete. Results saved to benchmark_results.csv and benchmark_plot.png")

if __name__ == '__main__':
    run_benchmark()

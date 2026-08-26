import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

def get_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > num:
                break
            if num % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
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
        self.in_features = in_features
        self.out_features = out_features
        
        # Start indices in the middle (close to 0)
        # Add slight randomness so they aren't all exactly identical
        center = LUT_SIZE // 2
        # Initialize randomly across the entire LUT
        self.idx = nn.Parameter(torch.randint(0, LUT_SIZE, (out_features, in_features), dtype=torch.int32), requires_grad=False)
        self.bias_idx = nn.Parameter(torch.randint(0, LUT_SIZE, (out_features,), dtype=torch.int32), requires_grad=False)
        
        self.vote_buffer = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_vote_buffer = torch.zeros((out_features,), dtype=torch.float32)

    def get_weight(self, idx_perturbation=None):
        if idx_perturbation is None:
            w_idx = self.idx
        else:
            w_idx = torch.clamp(self.idx + idx_perturbation, 0, LUT_SIZE - 1)
        return LUT[w_idx]

    def get_bias(self, idx_perturbation=None):
        if idx_perturbation is None:
            b_idx = self.bias_idx
        else:
            b_idx = torch.clamp(self.bias_idx + idx_perturbation, 0, LUT_SIZE - 1)
        return LUT[b_idx]

    def forward(self, x, weight_perturb=None, bias_perturb=None):
        w = self.get_weight(weight_perturb)
        b = self.get_bias(bias_perturb)
        return nn.functional.linear(x, w, b)


class CartPoleAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # CartPole has 4 observations and 2 actions, single linear layer is enough to solve it
        self.layer1 = PrimeLinear(4, 2)
        
    def forward(self, x, perturbations=None):
        if perturbations is None:
            w1_p, b1_p = None, None
        else:
            w1_p, b1_p = perturbations
            
        x = self.layer1(x, w1_p, b1_p)
        return x

def run_episode(env, agent, perturbations=None):
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(obs_t, perturbations)
            action = torch.argmax(q_values, dim=1).item()
            
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
    return total_reward

def train():
    env = gym.make('CartPole-v1')
    master = CartPoleAgent()
    
    pop_size = 64
    generations = 200
    
    threshold = 10.0  # Initial threshold
    target_flip_rate = 0.1 # 10% flip rate
    
    print("Starting PRIME Genetic Evolution on CartPole...")
    
    for gen in range(generations):
        t0 = time.time()
        # Generate perturbations for the population
        pop_perturbations = []
        for _ in range(pop_size):
            w1_p = torch.randint(-2000, 2001, master.layer1.idx.shape, dtype=torch.int32)
            b1_p = torch.randint(-2000, 2001, master.layer1.bias_idx.shape, dtype=torch.int32)
            pop_perturbations.append((w1_p, b1_p))
            
        # Evaluate population
        fitnesses = []
        for p in pop_perturbations:
            fit = run_episode(env, master, p)
            fitnesses.append(fit)
            
        fitnesses = np.array(fitnesses)
        mean_fit = fitnesses.mean()
        std_fit = fitnesses.std() + 1e-8
        advantages = (fitnesses - mean_fit) / std_fit
        
        # Accumulate votes
        for i, (w1_p, b1_p) in enumerate(pop_perturbations):
            adv = advantages[i]
            # Since perturbation is in [-2000, 2000], divide to get direction vector
            master.layer1.vote_buffer += (w1_p.float() / 2000.0) * adv
            master.layer1.bias_vote_buffer += (b1_p.float() / 2000.0) * adv
            
        # Apply supermajority gate
        total_params = 0
        total_flips = 0
        
        for layer in [master.layer1]:
            for param_idx, vote_buf in [(layer.idx, layer.vote_buffer), (layer.bias_idx, layer.bias_vote_buffer)]:
                flips = (torch.abs(vote_buf) > threshold).int()
                signs = torch.sign(vote_buf).int()
                
                # Apply flips (step by 1000 indices at a time)
                param_idx.data.add_(flips * signs * 1000)
                param_idx.data.clamp_(0, LUT_SIZE - 1)
                
                # Reset vote buffer for flipped parameters
                vote_buf[flips > 0] = 0
                
                total_flips += flips.sum().item()
                total_params += param_idx.numel()
                
                # Decay votes to forget stale information (momentum)
                vote_buf.mul_(0.9)
                
        # PI Homeostasis Controller (simplified)
        actual_flip_rate = total_flips / total_params
        error = actual_flip_rate - target_flip_rate
        # Increase threshold if flipping too much, decrease if too little
        threshold = max(5.0, threshold * (1.0 + error * 10.0))
        
        master_fit = run_episode(env, master)
        t1 = time.time()
        
        print(f"Gen {gen:3d} | Mean Fit: {mean_fit:6.1f} | Master Fit: {master_fit:6.1f} | Flips: {total_flips:4d} ({actual_flip_rate*100:.2f}%) | Thresh: {threshold:.1f} | Time: {t1-t0:.2f}s")
        
        if master_fit >= 500:
            # Verify it's consistently balancing
            test_fits = [run_episode(env, master) for _ in range(5)]
            if np.mean(test_fits) >= 500:
                print(f"\\nSolved! Consistently balancing at max steps.")
                break
                
    env.close()

if __name__ == "__main__":
    train()

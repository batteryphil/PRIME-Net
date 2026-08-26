import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import pygame
import sys

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

# --- Training Loop with Grid Renderer ---
def train():
    pygame.init()
    # 8x8 Grid. Each thumbnail is 150x100.
    THUMB_W, THUMB_H = 150, 100
    GRID_COLS, GRID_ROWS = 8, 8
    screen = pygame.display.set_mode((THUMB_W * GRID_COLS, THUMB_H * GRID_ROWS))
    pygame.display.set_caption("PRIME Genetic Evolution - 64 Parallel Agents")
    
    pop_size = 64
    generations = 300
    
    envs = [gym.make('LunarLander-v3', render_mode='rgb_array') for _ in range(pop_size)]
    master = LunarAgent()
    
    # Golden Configuration for Multi-Layer
    mut_scale = 1000
    step_size = 1000
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME Genetic Evolution Grid Renderer...")
    
    for gen in range(generations):
        t0 = time.time()
        
        # Generate perturbations
        pop_p = []
        for _ in range(pop_size):
            w1_p = torch.randint(-mut_scale, mut_scale+1, master.layer1.idx.shape, dtype=torch.int32)
            b1_p = torch.randint(-mut_scale, mut_scale+1, master.layer1.bias_idx.shape, dtype=torch.int32)
            w2_p = torch.randint(-mut_scale, mut_scale+1, master.layer2.idx.shape, dtype=torch.int32)
            b2_p = torch.randint(-mut_scale, mut_scale+1, master.layer2.bias_idx.shape, dtype=torch.int32)
            pop_p.append((w1_p, b1_p, w2_p, b2_p))
            
        # Reset all environments
        obses = [e.reset()[0] for e in envs]
        dones = [False] * pop_size
        total_rewards = np.zeros(pop_size)
        
        # Evaluate concurrently and render
        while not all(dones):
            # Process PyGame events so window doesn't freeze
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                    
            for i in range(pop_size):
                if dones[i]: continue
                
                obs_t = torch.tensor(obses[i], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q = master(obs_t, pop_p[i])
                    action = torch.argmax(q, dim=1).item()
                    
                obses[i], reward, terminated, truncated, _ = envs[i].step(action)
                total_rewards[i] += reward
                if terminated or truncated:
                    dones[i] = True
                    
                # Extract and draw frame
                frame = envs[i].render()
                if frame is not None:
                    # Convert to PyGame surface (transpose HWC to WHC for pygame)
                    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, (THUMB_W, THUMB_H))
                    
                    row = i // GRID_COLS
                    col = i % GRID_COLS
                    screen.blit(surf, (col * THUMB_W, row * THUMB_H))
                    
            pygame.display.flip()
            
        fits = total_rewards
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        # Accumulate Votes
        for i, (w1_p, b1_p, w2_p, b2_p) in enumerate(pop_p):
            master.layer1.vote_buffer += (w1_p.float() / float(mut_scale)) * advs[i]
            master.layer1.bias_vote_buffer += (b1_p.float() / float(mut_scale)) * advs[i]
            master.layer2.vote_buffer += (w2_p.float() / float(mut_scale)) * advs[i]
            master.layer2.bias_vote_buffer += (b2_p.float() / float(mut_scale)) * advs[i]
            
        # Apply Supermajority Gate
        total_flips, total_params = 0, 0
        layers = [master.layer1, master.layer2]
        
        for layer in layers:
            for p_idx, v_buf in [(layer.idx, layer.vote_buffer), (layer.bias_idx, layer.bias_vote_buffer)]:
                flips = (torch.abs(v_buf) > threshold).int()
                signs = torch.sign(v_buf).int()
                p_idx.data.add_(flips * signs * step_size).clamp_(0, LUT_SIZE - 1)
                v_buf[flips > 0] = 0
                v_buf.mul_(0.9)
                total_flips += flips.sum().item()
                total_params += p_idx.numel()
                
        rate = total_flips / total_params
        threshold = max(5.0, threshold * (1.0 + (rate - target_flip_rate)))
        
        print(f"Gen {gen:3d} | Mean Fit: {fits.mean():7.1f} | Max Fit: {fits.max():7.1f} | Flips: {total_flips:4d} ({rate*100:5.2f}%) | Thresh: {threshold:5.1f} | Time: {time.time()-t0:.2f}s")
        
    for e in envs: e.close()
    pygame.quit()

if __name__ == '__main__':
    train()

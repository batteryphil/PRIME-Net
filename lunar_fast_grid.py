import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import pygame
import sys
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
    return torch.tensor(lut, dtype=torch.float32).to(device)

LUT = build_prime_lut(size=65536)
LUT_SIZE = len(LUT)

class PrimeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.idx = nn.Parameter(torch.randint(0, LUT_SIZE, (out_features, in_features), dtype=torch.int32, device=device), requires_grad=False)
        self.bias_idx = nn.Parameter(torch.randint(0, LUT_SIZE, (out_features,), dtype=torch.int32, device=device), requires_grad=False)
        self.vote_buffer = torch.zeros((out_features, in_features), dtype=torch.float32, device=device)
        self.bias_vote_buffer = torch.zeros((out_features,), dtype=torch.float32, device=device)

    def forward(self, x, w_p=None, b_p=None):
        if w_p is None:
            w = LUT[self.idx]
            b = LUT[self.bias_idx]
            return F.linear(x, w, b)
        else:
            # Batched forward pass for population
            w_idx = torch.clamp(self.idx.unsqueeze(0) + w_p, 0, LUT_SIZE - 1)
            b_idx = torch.clamp(self.bias_idx.unsqueeze(0) + b_p, 0, LUT_SIZE - 1)
            w = LUT[w_idx] # (Batch, Out, In)
            b = LUT[b_idx] # (Batch, Out)
            
            w_T = w.transpose(1, 2) # (Batch, In, Out)
            out = torch.bmm(x.unsqueeze(1), w_T).squeeze(1) + b # (Batch, Out)
            return out

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

# --- Fast Training Loop with GPU Renderer ---
def train():
    pygame.init()
    THUMB_W, THUMB_H = 150, 100
    GRID_COLS, GRID_ROWS = 8, 8
    screen = pygame.display.set_mode((THUMB_W * GRID_COLS, THUMB_H * GRID_ROWS))
    pygame.display.set_caption("PRIME ES - GPU Accelerated AsyncVectorEnv")
    
    pop_size = 64
    generations = 300
    
    # Vectorized Envs (Multiprocessing)
    envs = gym.make_vec('LunarLander-v3', num_envs=pop_size, vectorization_mode='async', render_mode='rgb_array')
    master = LunarAgent().to(device)
    
    # Golden Configuration
    mut_scale = 1000
    step_size = 1000
    target_flip_rate = 0.05
    threshold = 10.0
    
    print("Starting PRIME Fast GPU Grid Renderer...")
    
    for gen in range(generations):
        t0 = time.time()
        
        # Generate batch perturbations on GPU
        w1_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer1.idx.shape), dtype=torch.int32, device=device)
        b1_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer1.bias_idx.shape), dtype=torch.int32, device=device)
        w2_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer2.idx.shape), dtype=torch.int32, device=device)
        b2_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer2.bias_idx.shape), dtype=torch.int32, device=device)
        
        pop_p = (w1_p, b1_p, w2_p, b2_p)
        
        obs, info = envs.reset()
        dones = np.zeros(pop_size, dtype=bool)
        total_rewards = np.zeros(pop_size, dtype=np.float32)
        
        # Fast Vectorized Evaluation
        while not dones.all():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    envs.close()
                    pygame.quit()
                    sys.exit()
            
            # Forward pass all 64 agents simultaneously on GPU
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q = master(obs_t, pop_p)
                actions = torch.argmax(q, dim=1).cpu().numpy()
                
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            # Accumulate rewards for those still running
            total_rewards += rewards * (~dones)
            dones = dones | terminated | truncated
            
            # Fast GPU Rendering Pipeline
            frames = envs.render() # Tuple of (400, 600, 3) arrays
            if frames and len(frames) == pop_size and frames[0] is not None:
                frames_np = np.stack(frames) # (64, 400, 600, 3)
                # To GPU, format (N, C, H, W)
                frames_t = torch.from_numpy(frames_np).to(device).permute(0, 3, 1, 2).float()
                # Fast GPU resize
                resized_t = F.interpolate(frames_t, size=(THUMB_H, THUMB_W))
                # Back to CPU for PyGame, format (N, W, H, C) for blitting
                resized_np = resized_t.permute(0, 3, 2, 1).byte().cpu().numpy()
                
                # Blit grid
                for i in range(pop_size):
                    row, col = i // GRID_COLS, i % GRID_COLS
                    surf = pygame.surfarray.make_surface(resized_np[i])
                    screen.blit(surf, (col * THUMB_W, row * THUMB_H))
                pygame.display.flip()
                
        fits = torch.tensor(total_rewards, dtype=torch.float32, device=device)
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        # Batched Vote Accumulation
        master.layer1.vote_buffer += torch.sum((w1_p.float() / float(mut_scale)) * advs.view(-1, 1, 1), dim=0)
        master.layer1.bias_vote_buffer += torch.sum((b1_p.float() / float(mut_scale)) * advs.view(-1, 1), dim=0)
        master.layer2.vote_buffer += torch.sum((w2_p.float() / float(mut_scale)) * advs.view(-1, 1, 1), dim=0)
        master.layer2.bias_vote_buffer += torch.sum((b2_p.float() / float(mut_scale)) * advs.view(-1, 1), dim=0)
        
        # Apply Supermajority Gate
        total_flips, total_params = 0, 0
        for layer in [master.layer1, master.layer2]:
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
        
        print(f"Gen {gen:3d} | Mean Fit: {fits.mean().item():7.1f} | Max Fit: {fits.max().item():7.1f} | Flips: {total_flips:4d} ({rate*100:5.2f}%) | Thresh: {threshold:5.1f} | Time: {time.time()-t0:.2f}s")
        
    envs.close()
    pygame.quit()

if __name__ == '__main__':
    train()

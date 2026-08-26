import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import pygame
import sys
import math
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
        
        self.step_sizes = torch.ones((out_features, in_features), dtype=torch.int32, device=device)
        self.bias_step_sizes = torch.ones((out_features,), dtype=torch.int32, device=device)
        self.last_signs = torch.zeros((out_features, in_features), dtype=torch.int32, device=device)
        self.bias_last_signs = torch.zeros((out_features,), dtype=torch.int32, device=device)

    def forward(self, x, w_p=None, b_p=None):
        if w_p is None:
            w = LUT[self.idx]
            b = LUT[self.bias_idx]
            return F.linear(x, w, b)
        else:
            w_idx = torch.clamp(self.idx.unsqueeze(0) + w_p, 0, LUT_SIZE - 1)
            b_idx = torch.clamp(self.bias_idx.unsqueeze(0) + b_p, 0, LUT_SIZE - 1)
            w = LUT[w_idx]
            b = LUT[b_idx]
            w_T = w.transpose(1, 2)
            out = torch.bmm(x.unsqueeze(1), w_T).squeeze(1) + b
            return out

class LunarAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PrimeLinear(8, 128)
        self.layer2 = PrimeLinear(128, 4)
        
    def forward(self, x, p=None):
        if p is None:
            x = torch.relu(self.layer1(x))
            return self.layer2(x)
        else:
            w1_p, b1_p, w2_p, b2_p = p
            x = torch.relu(self.layer1(x, w1_p, b1_p))
            return self.layer2(x, w2_p, b2_p)

# --- Phase 4: Inverted Cone Curriculum ---
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.curriculum_factor = 0.0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            lander = self.unwrapped.lander
            if lander is not None:
                helipad_y = getattr(self.unwrapped, 'helipad_y', 0.0)
                target_y = helipad_y + 10.0 + (self.curriculum_factor * 10.0)
                lander.position = (lander.position.x * self.curriculum_factor, target_y)
                lander.linearVelocity = (lander.linearVelocity[0] * self.curriculum_factor, lander.linearVelocity[1] * self.curriculum_factor)
                obs, _, _, _, _ = self.unwrapped.step(np.array([0,0]))
        except:
            pass
        return obs, info

def make_env():
    env = gym.make('LunarLander-v3')
    return CurriculumWrapper(env)

# --- Ghost Overlay Renderer Loop ---
def train():
    pygame.init()
    SCREEN_W, SCREEN_H = 800, 600
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("PRIME ES - Ghost Overlay Swarm")
    clock = pygame.time.Clock()
    
    # Pre-render surfaces
    ghost_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
    ghost_surf.fill((100, 150, 255, 60)) # Semi-transparent blue
    master_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
    master_surf.fill((255, 50, 50, 255)) # Solid red
    
    font = pygame.font.SysFont(None, 36)
    
    pop_size = 512
    generations = 2000
    best_score = -float('inf')
    
    # We do NOT request rgb_array. Just raw physics math.
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(pop_size)])
    master = LunarAgent().to(device)
    
    mut_scale = 1000
    step_size = 1000
    target_flip_rate = 0.05
    threshold = 10.0
    
    # Initialize log file
    with open('progress_log.txt', 'w') as f:
        f.write("Generation,Mean_Fit,Master_Fit,Max_Fit,Flips,Flip_Rate,Threshold,Time\n")
    
    print("Starting PRIME Ghost Swarm (Speed Optimized)...")
    
    for gen in range(generations):
        t0 = time.time()
        
        # Update Curriculum Factor
        c_factor = min(1.0, gen / 1000.0)
        envs.set_attr('curriculum_factor', c_factor)
        
        w1_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer1.idx.shape), dtype=torch.int32, device=device)
        b1_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer1.bias_idx.shape), dtype=torch.int32, device=device)
        w2_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer2.idx.shape), dtype=torch.int32, device=device)
        b2_p = torch.randint(-mut_scale, mut_scale+1, (pop_size, *master.layer2.bias_idx.shape), dtype=torch.int32, device=device)
        
        # Mutation Sparsity (20% Mask)
        w1_mask = (torch.rand(w1_p.shape, device=device) < 0.20).int()
        b1_mask = (torch.rand(b1_p.shape, device=device) < 0.20).int()
        w2_mask = (torch.rand(w2_p.shape, device=device) < 0.20).int()
        b2_mask = (torch.rand(b2_p.shape, device=device) < 0.20).int()
        
        w1_p *= w1_mask
        b1_p *= b1_mask
        w2_p *= w2_mask
        b2_p *= b2_mask
        
        # Agent 0 is the true Master
        w1_p[0] = 0; b1_p[0] = 0; w2_p[0] = 0; b2_p[0] = 0
        
        pop_p = (w1_p, b1_p, w2_p, b2_p)
        
        obs, info = envs.reset()
        dones = np.zeros(pop_size, dtype=bool)
        total_rewards = np.zeros(pop_size, dtype=np.float32)
        steps = 0
        
        while not dones.all():
            steps += 1
            if steps > 350:
                break
                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    envs.close()
                    pygame.quit()
                    sys.exit()
                    
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q = master(obs_t, pop_p)
                actions = torch.argmax(q, dim=1).cpu().numpy()
                
            obs, rewards, terminated, truncated, _ = envs.step(actions)
            
            # --- CUSTOM REWARD SHAPING ---
            # 1. Angle Penalty: Punish landers that tilt too far left/right to force upright orientation.
            angle_penalty = np.abs(obs[:, 4]) * 5.0
            rewards -= angle_penalty
            
            # -----------------------------
            
            total_rewards += rewards * (~dones)
            dones = dones | terminated | truncated
            
            # --- Ghost Rendering ---
            screen.fill((10, 10, 20)) # Dark space background
            pygame.draw.line(screen, (200, 200, 200), (0, 550), (800, 550), 2) # Ground
            pygame.draw.line(screen, (0, 255, 0), (350, 550), (450, 550), 5) # Landing Pad
            
            # Draw workers first (so master is on top)
            for i in range(1, pop_size):
                if not dones[i]:
                    x = int(400 + obs[i][0] * 350)
                    y = int(500 - obs[i][1] * 350)
                    angle = -obs[i][4] * 180.0 / math.pi
                    rotated = pygame.transform.rotate(ghost_surf, angle)
                    rect = rotated.get_rect(center=(x, y))
                    screen.blit(rotated, rect.topleft)
                    
            # Draw Master (Agent 0)
            if not dones[0]:
                x = int(400 + obs[0][0] * 350)
                y = int(500 - obs[0][1] * 350)
                angle = -obs[0][4] * 180.0 / math.pi
                rotated = pygame.transform.rotate(master_surf, angle)
                rect = rotated.get_rect(center=(x, y))
                screen.blit(rotated, rect.topleft)
                
            # Draw UI
            if best_score != -float('inf'):
                score_text = font.render(f"Top Score: {best_score:.1f}", True, (255, 255, 255))
                screen.blit(score_text, (10, 10))
            gen_text = font.render(f"Generation: {gen}", True, (255, 255, 255))
            screen.blit(gen_text, (10, 40))
                
            pygame.display.flip()
            # Removed clock.tick(60) to unlock FPS and run as fast as possible
            
        fits = torch.tensor(total_rewards, dtype=torch.float32, device=device)
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        max_fit = fits.max().item()
        mean_fit = fits.mean().item()
        master_fit = total_rewards[0]
        
        master.layer1.vote_buffer += torch.sum((w1_p.float() / float(mut_scale)) * advs.view(-1, 1, 1), dim=0)
        master.layer1.bias_vote_buffer += torch.sum((b1_p.float() / float(mut_scale)) * advs.view(-1, 1), dim=0)
        master.layer2.vote_buffer += torch.sum((w2_p.float() / float(mut_scale)) * advs.view(-1, 1, 1), dim=0)
        master.layer2.bias_vote_buffer += torch.sum((b2_p.float() / float(mut_scale)) * advs.view(-1, 1), dim=0)
        
        total_flips, total_params = 0, 0
        gamma = 0.9
        std_fit = fits.std().item()
        if std_fit < 5.0:
            gamma = 0.7  # Adaptive Memory Decay
        elif std_fit > 50.0:
            gamma = 0.95
            
        gamma = max(0.70, min(0.95, gamma))
            
        for layer in [master.layer1, master.layer2]:
            for p_idx, v_buf, steps_buf, signs_buf in [(layer.idx, layer.vote_buffer, layer.step_sizes, layer.last_signs), 
                                                       (layer.bias_idx, layer.bias_vote_buffer, layer.bias_step_sizes, layer.bias_last_signs)]:
                current_signs = torch.sign(v_buf).int()
                
                # Rprop Momentum anchored to continuous gradient
                same_sign = (current_signs == signs_buf) & (current_signs != 0)
                diff_sign = (current_signs != signs_buf) & (current_signs != 0)
                
                steps_buf[same_sign] = torch.clamp(steps_buf[same_sign] * 2, max=8)
                steps_buf[diff_sign] = torch.clamp(steps_buf[diff_sign] // 2, min=1)
                
                # Update tracker with continuous sign
                signs_buf[current_signs != 0] = current_signs[current_signs != 0]
                
                # Execute flips based on threshold
                flips = (torch.abs(v_buf) > threshold).int()
                p_idx.data.add_(flips * current_signs * steps_buf).clamp_(0, LUT_SIZE - 1)
                
                v_buf[flips > 0] = 0
                v_buf.mul_(gamma)
                total_flips += flips.sum().item()
                total_params += p_idx.numel()
                
        rate = total_flips / total_params
        threshold = max(5.0, threshold * (1.0 + (rate - target_flip_rate)))
        
        best_score = max(best_score, max_fit)
        
        log_line = f"{gen},{mean_fit:.1f},{master_fit:.1f},{max_fit:.1f},{total_flips},{rate*100:.2f},{threshold:.1f},{time.time()-t0:.2f}\n"
        with open('progress_log.txt', 'a') as f:
            f.write(log_line)
            
        print(f"Gen {gen:3d} | Mean Fit: {mean_fit:7.1f} | Master Fit: {master_fit:7.1f} | Max Fit: {max_fit:7.1f} | Flips: {total_flips:4d} ({rate*100:5.2f}%) | Thresh: {threshold:5.1f} | Time: {time.time()-t0:.2f}s")
        
        if master_fit >= 200:
            print(f"LunarLander SOLVED in {gen} generations!")
            break
            
    envs.close()
    pygame.quit()

if __name__ == '__main__':
    train()

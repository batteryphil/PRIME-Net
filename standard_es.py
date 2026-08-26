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

class LunarAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 32)
        self.layer2 = nn.Linear(32, 4)
        
    def forward(self, x, p=None):
        if p is None:
            x = torch.relu(self.layer1(x))
            return self.layer2(x)
        else:
            w1_p, b1_p, w2_p, b2_p = p
            # w1_p shape: (pop_size, 32, 8)
            # x shape: (pop_size, 8) -> x_u shape: (pop_size, 1, 8)
            x_u = x.unsqueeze(1)
            
            # W1 shape: (pop_size, 32, 8) -> W1_T shape: (pop_size, 8, 32)
            W1 = self.layer1.weight.unsqueeze(0) + w1_p
            b1 = self.layer1.bias.unsqueeze(0) + b1_p
            
            out1 = torch.relu(torch.bmm(x_u, W1.transpose(1, 2)).squeeze(1) + b1)
            
            W2 = self.layer2.weight.unsqueeze(0) + w2_p
            b2 = self.layer2.bias.unsqueeze(0) + b2_p
            
            out2 = torch.bmm(out1.unsqueeze(1), W2.transpose(1, 2)).squeeze(1) + b2
            return out2

def train():
    pygame.init()
    SCREEN_W, SCREEN_H = 800, 600
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Continuous ES - Ghost Overlay Swarm")
    
    ghost_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
    ghost_surf.fill((100, 150, 255, 60)) 
    master_surf = pygame.Surface((40, 20), pygame.SRCALPHA)
    master_surf.fill((255, 50, 50, 255)) 
    
    font = pygame.font.SysFont(None, 36)
    
    pop_size = 256
    generations = 300
    best_score = -float('inf')
    
    envs = gym.make_vec('LunarLander-v3', num_envs=pop_size, vectorization_mode='async')
    master = LunarAgent().to(device)
    
    noise_std = 0.1
    learning_rate = 0.05
    
    with open('standard_progress_log.txt', 'w') as f:
        f.write("Generation,Mean_Fit,Master_Fit,Max_Fit,Time\n")
    
    print("Starting Continuous ES Ghost Swarm...")
    
    for gen in range(generations):
        t0 = time.time()
        
        w1_p = torch.randn(pop_size, 32, 8, device=device) * noise_std
        b1_p = torch.randn(pop_size, 32, device=device) * noise_std
        w2_p = torch.randn(pop_size, 4, 32, device=device) * noise_std
        b2_p = torch.randn(pop_size, 4, device=device) * noise_std
        
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
            
            angle_penalty = np.abs(obs[:, 4]) * 5.0
            rewards -= angle_penalty
            
            if steps > 350:
                rewards -= 100 * (~dones)
                
            total_rewards += rewards * (~dones)
            dones = dones | terminated | truncated
            
            screen.fill((10, 10, 20))
            pygame.draw.line(screen, (200, 200, 200), (0, 550), (800, 550), 2)
            pygame.draw.line(screen, (0, 255, 0), (350, 550), (450, 550), 5)
            
            for i in range(1, pop_size):
                if not dones[i]:
                    x = int(400 + obs[i][0] * 350)
                    y = int(500 - obs[i][1] * 350)
                    angle = -obs[i][4] * 180.0 / math.pi
                    rotated = pygame.transform.rotate(ghost_surf, angle)
                    rect = rotated.get_rect(center=(x, y))
                    screen.blit(rotated, rect.topleft)
                    
            if not dones[0]:
                x = int(400 + obs[0][0] * 350)
                y = int(500 - obs[0][1] * 350)
                angle = -obs[0][4] * 180.0 / math.pi
                rotated = pygame.transform.rotate(master_surf, angle)
                rect = rotated.get_rect(center=(x, y))
                screen.blit(rotated, rect.topleft)
                
            if best_score != -float('inf'):
                score_text = font.render(f"Top Score: {best_score:.1f}", True, (255, 255, 255))
                screen.blit(score_text, (10, 10))
            gen_text = font.render(f"Generation: {gen}", True, (255, 255, 255))
            screen.blit(gen_text, (10, 40))
                
            pygame.display.flip()
            
        fits = torch.tensor(total_rewards, dtype=torch.float32, device=device)
        advs = (fits - fits.mean()) / (fits.std() + 1e-8)
        
        with torch.no_grad():
            w1_grad = torch.sum(w1_p * advs.view(-1, 1, 1), dim=0) / (pop_size * noise_std)
            b1_grad = torch.sum(b1_p * advs.view(-1, 1), dim=0) / (pop_size * noise_std)
            w2_grad = torch.sum(w2_p * advs.view(-1, 1, 1), dim=0) / (pop_size * noise_std)
            b2_grad = torch.sum(b2_p * advs.view(-1, 1), dim=0) / (pop_size * noise_std)
            
            master.layer1.weight.data += learning_rate * w1_grad
            master.layer1.bias.data += learning_rate * b1_grad
            master.layer2.weight.data += learning_rate * w2_grad
            master.layer2.bias.data += learning_rate * b2_grad
        
        master_fit = total_rewards[0]
        max_fit = fits.max().item()
        mean_fit = fits.mean().item()
        
        best_score = max(best_score, max_fit)
        
        log_line = f"{gen},{mean_fit:.1f},{master_fit:.1f},{max_fit:.1f},{time.time()-t0:.2f}\n"
        with open('standard_progress_log.txt', 'a') as f:
            f.write(log_line)
            
        print(f"Gen {gen:3d} | Mean: {mean_fit:7.1f} | Master: {master_fit:7.1f} | Max: {max_fit:7.1f} | Time: {time.time()-t0:.2f}s")
        
    envs.close()
    pygame.quit()

if __name__ == '__main__':
    train()

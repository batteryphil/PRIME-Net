import gymnasium as gym
import numpy as np
import sympy as sp
from stable_baselines3 import PPO
from prime_core import run_prime_engine
import imageio
import os

def collect_expert_telemetry():
    print("[*] Training PPO Expert on LunarLanderContinuous-v3...")
    env = gym.make("LunarLanderContinuous-v3")
    
    # Train for 500k steps to ensure it becomes a Master Pilot (soft hovering)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=500000)
    print("[*] Expert Trained.")
    
    # Collect Telemetry
    print("[*] Collecting 50 episodes of Expert Telemetry...")
    X_data = []
    Y_main = []
    Y_side = []
    
    for _ in range(50):
        obs, _ = env.reset()
        done = False
        while not done:
            # Get action from expert
            action, _ = model.predict(obs, deterministic=True)
            
            X_data.append(obs)
            Y_main.append(action[0])
            Y_side.append(action[1])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
    env.close()
    
    X = np.array(X_data)
    Y1 = np.array(Y_main)
    Y2 = np.array(Y_side)
    
    print(f"[*] Collected {len(X)} telemetry frames.")
    return X, Y1, Y2

def run_lunar_lander_test():
    print("==========================================")
    print("PRIME-Net: BEHAVIORAL CLONING (LUNAR LANDER)")
    print("==========================================")
    
    X, Y1, Y2 = collect_expert_telemetry()
    
    # We will just run PRIME directly on the unnormalized data to get the purest equation
    # Wait, AFPO in prime_core expects data to be somewhat well-scaled.
    # Let's normalize X, but keep Y bounded in [-1, 1] as the environment expects
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    engine_config = {
        'pop_size': 512,
        'seq_len': 31,
        'macro_seq_len': 7,
        'max_generations': 2000,
        'timeout_sec': 60.0 # 1 minute per equation
    }
    
    print("\n[*] Launching PRIME-Net to compress Main Engine neural policy into an equation...")
    res_main = run_prime_engine(X_norm, Y1, X_norm, Y1, **engine_config)
    main_eq = res_main['best_sympy']
    print(f"[+] Main Engine Equation: {main_eq}")
    
    print("\n[*] Launching PRIME-Net to compress Side Engine neural policy into an equation...")
    res_side = run_prime_engine(X_norm, Y2, X_norm, Y2, **engine_config)
    side_eq = res_side['best_sympy']
    print(f"[+] Side Engine Equation: {side_eq}")
    
    print("\n[*] Unplugging the Neural Network...")
    print("[*] Mathematical Pilot Taking Over. Rendering flight to video...")
    
    # Lambdify the symbolic equations for fast execution
    try:
        expr_main = sp.sympify(main_eq)
        expr_side = sp.sympify(side_eq)
    except Exception as e:
        print(f"[!] Failed to parse equations: {e}")
        return
        
    symbols = [sp.Symbol(f'X{i+1}') for i in range(8)]
    f_main = sp.lambdify(symbols, expr_main, 'numpy')
    f_side = sp.lambdify(symbols, expr_side, 'numpy')
    
    render_env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    obs, _ = render_env.reset(seed=42)
    
    frames = []
    total_reward = 0
    done = False
    
    while not done:
        frame = render_env.render()
        frames.append(frame)
        
        # Normalize observation
        obs_norm = (obs - X_mean) / X_std
        
        # Evaluate equations
        try:
            m_thrust = float(f_main(*obs_norm))
            s_thrust = float(f_side(*obs_norm))
        except Exception:
            m_thrust = 0.0
            s_thrust = 0.0
            
        # Clip to environment bounds [-1, 1]
        m_thrust = np.clip(m_thrust, -1.0, 1.0)
        s_thrust = np.clip(s_thrust, -1.0, 1.0)
        
        action = np.array([m_thrust, s_thrust])
        obs, reward, terminated, truncated, _ = render_env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    render_env.close()
    
    print(f"[*] Mathematical Landing Complete! Total Reward: {total_reward:.2f}")
    
    # Save video
    vid_path = "/home/phil/.gemini/antigravity/scratch/prime_neat/math_lander.mp4"
    imageio.mimsave(vid_path, frames, fps=30)
    print(f"[*] Video saved to {vid_path}")
    print("==========================================")

if __name__ == "__main__":
    run_lunar_lander_test()

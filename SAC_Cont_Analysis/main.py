#!/usr/bin/env python3
import os
import json
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from SAC import SAC_agent
import psutil
from datetime import datetime
from utils import evaluate_policy

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_ram_usage():
    """Get RAM usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def get_cpu_usage():
    """Get CPU usage percentage"""
    return psutil.cpu_percent()

def train(env_name, env_index, max_steps, device='cpu', write=True, render=False, 
          load_model=False, model_index=None, record_video=True, video_folder='videos',
          seed=0, save_interval=10000, eval_interval=1000, gamma=0.99, net_width=256,
          a_lr=3e-4, c_lr=3e-4, batch_size=256, random_steps=10000):
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment with video recording if enabled
    render_mode = 'rgb_array' if record_video else None
    env = gym.make(env_name, render_mode=render_mode)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Create video folder structure if recording is enabled
    if record_video:
        env_video_folder = os.path.join(video_folder, env_name)
        os.makedirs(env_video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, 
                                     video_folder=env_video_folder,
                                     episode_trigger=lambda x: x % 300 == 0)  # Record every 300th episode
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize agent
    agent = SAC_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        dvc=device,
        gamma=gamma,
        net_width=net_width,
        a_lr=a_lr,
        c_lr=c_lr,
        batch_size=batch_size
    )
    
    # Load model if specified
    if load_model and model_index is not None:
        agent.load(env_name, model_index)
    
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'ram_usage_gb': [],  # List to store RAM usage at each episode
        'cpu_usage_percent': [],  # List to store CPU usage at each episode
        'total_steps': 0,
        'training_start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training_time_min': 0,
        'steps_to_convergence': None,
        'time_to_convergence': None,
        'collapse_events': 0,
        'sample_efficiency': [],
        'robustness_drop_percent': 0.0,
        'eval_rewards': []  # Track evaluation rewards
    }
    
    # Training loop
    total_steps = 0
    episode = 0
    start_time = time.time()
    best_reward = float('-inf')
    convergence_threshold = 0.95  # 95% of max possible reward
    convergence_window = 10  # Number of episodes to check for convergence
    collapse_threshold = 0.5  # 50% drop in performance
    last_avg_reward = None
    
    while total_steps < max_steps:
        state, info = env.reset()
        episode_reward = 0
        done = False
        episode_steps = 0
        
        while not done and total_steps < max_steps:
            # Select action
            if total_steps < random_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, terminated)
            
            # Train agent
            if total_steps >= random_steps:
                agent.train()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1
            
            # Record RAM usage at each step
            current_ram = get_ram_usage()
            metrics['ram_usage_gb'].append(current_ram)
            
            # Save model periodically
            if total_steps % save_interval == 0:
                agent.save(env_name, total_steps)
            
            # Evaluate periodically
            if total_steps % eval_interval == 0:
                eval_reward = evaluate_policy(env, agent)
                metrics['eval_rewards'].append(eval_reward)
                print(f"Step {total_steps}: Eval Reward = {eval_reward:.2f}")
                
                # Calculate sample efficiency
                efficiency = eval_reward / total_steps
                metrics['sample_efficiency'].append(efficiency)
                
                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    agent.save(env_name, 'best')
                
                # Check for convergence
                if metrics['steps_to_convergence'] is None:
                    recent_evals = metrics['eval_rewards'][-convergence_window:]
                    if len(recent_evals) >= convergence_window:
                        avg_reward = np.mean(recent_evals)
                        if avg_reward >= convergence_threshold * best_reward:
                            metrics['steps_to_convergence'] = total_steps
                            metrics['time_to_convergence'] = (time.time() - start_time) / 60
                
                # Check for collapse events
                if last_avg_reward is not None:
                    if eval_reward < last_avg_reward * (1 - collapse_threshold):
                        metrics['collapse_events'] += 1
                last_avg_reward = eval_reward
        
        # Record episode metrics
        metrics['episode_rewards'].append(episode_reward)
        
        # Record CPU usage only at episode boundaries
        current_cpu = get_cpu_usage()
        metrics['cpu_usage_percent'].append(current_cpu)
        
        # Update total metrics
        metrics['total_steps'] = total_steps
        metrics['training_time_min'] = (time.time() - start_time) / 60
        
        # Calculate robustness (drop from peak performance)
        if len(metrics['eval_rewards']) > 0:
            peak_reward = max(metrics['eval_rewards'])
            current_reward = metrics['eval_rewards'][-1]
            if peak_reward > 0:  # Avoid division by zero
                drop_percent = max(0, (peak_reward - current_reward) / peak_reward * 100)
                metrics['robustness_drop_percent'] = drop_percent
        
        # Save metrics periodically
        if total_steps % save_interval == 0:
            with open(f'results/metrics_{env_name}.json', 'w') as f:
                json.dump(metrics, f)
        
        episode += 1
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, RAM Usage = {current_ram:.2f} GB, CPU Usage = {current_cpu:.2f}%")
    
    # Save final metrics
    with open(f'results/metrics_{env_name}.json', 'w') as f:
        json.dump(metrics, f)
    
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dvc', type=str, default='cpu')
    parser.add_argument('--EnvIdex', type=int, default=0)
    parser.add_argument('--write', type=str2bool, default=True)
    parser.add_argument('--render', type=str2bool, default=False)
    parser.add_argument('--Loadmodel', type=str2bool, default=False)
    parser.add_argument('--ModelIdex', type=int, default=None)
    parser.add_argument('--record_video', type=str2bool, default=True)
    parser.add_argument('--video_folder', type=str, default='videos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--Max_train_steps', type=int, default=500000)
    parser.add_argument('--save_interval', type=int, default=10000)
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--net_width', type=int, default=256)
    parser.add_argument('--a_lr', type=float, default=3e-4)
    parser.add_argument('--c_lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_steps', type=int, default=10000)
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    os.makedirs(args.video_folder, exist_ok=True)
    
    # Map environment index to name
    env_names = [
        'Pendulum-v1',
        'LunarLanderContinuous-v3',
        'Humanoid-v5',
        'HalfCheetah-v4',
        'BipedalWalker-v3',
        'BipedalWalkerHardcore-v3'
    ]
    
    env_name = env_names[args.EnvIdex]
    
    # Start training
    train(
        env_name=env_name,
        env_index=args.EnvIdex,
        max_steps=args.Max_train_steps,
        device=args.dvc,
        write=args.write,
        render=args.render,
        load_model=args.Loadmodel,
        model_index=args.ModelIdex,
        record_video=args.record_video,
        video_folder=args.video_folder,
        seed=args.seed,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        gamma=args.gamma,
        net_width=args.net_width,
        a_lr=args.a_lr,
        c_lr=args.c_lr,
        batch_size=args.batch_size,
        random_steps=args.random_steps
    )

if __name__ == "__main__":
    main() 
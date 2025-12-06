"""
Transfer Learning: Train on easy students, transfer to hard students
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from environments.student_simulator_easy import EasyStudentSimulator
from environments.student_simulator_hard import HardStudentSimulator
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
import json

def train_transfer(num_episodes_source=250, num_episodes_target=250, save_dir='results/transfer'):
    os.makedirs(save_dir, exist_ok=True)
    
    # PHASE 1: Train on easy students
    print("="*70)
    print("PHASE 1: Training on EASY students (source task)")
    print("="*70)
    env_easy = EasyStudentSimulator(max_problems=20)
    agent = DQNAgent(state_size=6, action_size=6, learning_rate=0.0005, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    
    source_rewards, source_skills, source_accs = [], [], []
    
    for ep in range(num_episodes_source):
        state, _ = env_easy.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, term, trunc, info = env_easy.step(action)
            done = term or trunc
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            ep_reward += reward
        stats = env_easy.get_episode_stats()
        source_rewards.append(ep_reward)
        source_skills.append(stats['skill_gained'])
        source_accs.append(stats['accuracy'])
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{num_episodes_source}: Reward={np.mean(source_rewards[-50:]):.1f}")
    
    agent.save(f"{save_dir}/source_policy.pth")
    print("✅ Source task training complete!")
    
    # PHASE 2: Transfer to hard students (with transfer)
    print("\n" + "="*70)
    print("PHASE 2a: Transfer to HARD students (WITH pre-training)")
    print("="*70)
    env_hard = HardStudentSimulator(max_problems=20)
    
    transfer_rewards, transfer_skills, transfer_accs = [], [], []
    
    for ep in range(num_episodes_target):
        state, _ = env_hard.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, term, trunc, info = env_hard.step(action)
            done = term or trunc
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            ep_reward += reward
        stats = env_hard.get_episode_stats()
        transfer_rewards.append(ep_reward)
        transfer_skills.append(stats['skill_gained'])
        transfer_accs.append(stats['accuracy'])
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{num_episodes_target}: Reward={np.mean(transfer_rewards[-50:]):.1f}")
    
    agent.save(f"{save_dir}/transfer_policy.pth")
    print("✅ Transfer learning complete!")
    
    # PHASE 3: Train from scratch on hard students (NO transfer)
    print("\n" + "="*70)
    print("PHASE 2b: Train on HARD students (WITHOUT pre-training)")
    print("="*70)
    agent_scratch = DQNAgent(state_size=6, action_size=6, learning_rate=0.0005, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
    
    scratch_rewards, scratch_skills, scratch_accs = [], [], []
    
    for ep in range(num_episodes_target):
        state, _ = env_hard.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent_scratch.act(state, training=True)
            next_state, reward, term, trunc, info = env_hard.step(action)
            done = term or trunc
            agent_scratch.remember(state, action, reward, next_state, done)
            agent_scratch.replay()
            state = next_state
            ep_reward += reward
        stats = env_hard.get_episode_stats()
        scratch_rewards.append(ep_reward)
        scratch_skills.append(stats['skill_gained'])
        scratch_accs.append(stats['accuracy'])
        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{num_episodes_target}: Reward={np.mean(scratch_rewards[-50:]):.1f}")
    
    agent_scratch.save(f"{save_dir}/scratch_policy.pth")
    print("✅ Scratch training complete!")
    
    # Save all metrics
    metrics = {
        'source_rewards': source_rewards,
        'source_skills': source_skills,
        'source_accs': source_accs,
        'transfer_rewards': transfer_rewards,
        'transfer_skills': transfer_skills,
        'transfer_accs': transfer_accs,
        'scratch_rewards': scratch_rewards,
        'scratch_skills': scratch_skills,
        'scratch_accs': scratch_accs
    }
    
    with open(f"{save_dir}/transfer_metrics.json", 'w') as f:
        json.dump(metrics, f)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Transfer Learning: Pre-trained vs From Scratch', fontsize=16, fontweight='bold')
    
    smooth_fn = lambda d: list(np.convolve(d, np.ones(25)/25, mode='valid')) if len(d) > 25 else d
    
    axes[0].plot(smooth_fn(transfer_rewards), label='With Transfer', linewidth=2, color='green')
    axes[0].plot(smooth_fn(scratch_rewards), label='From Scratch', linewidth=2, color='red')
    axes[0].set_title('Rewards on Hard Students')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(smooth_fn(transfer_skills), label='With Transfer', linewidth=2, color='green')
    axes[1].plot(smooth_fn(scratch_skills), label='From Scratch', linewidth=2, color='red')
    axes[1].set_title('Skill Gains')
    axes[1].set_xlabel('Episode')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(smooth_fn(transfer_accs), label='With Transfer', linewidth=2, color='green')
    axes[2].plot(smooth_fn(scratch_accs), label='From Scratch', linewidth=2, color='red')
    axes[2].set_title('Accuracy')
    axes[2].set_xlabel('Episode')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/transfer_learning_comparison.png", dpi=300)
    print("📊 Plots saved!")
    
    print("\n" + "="*70)
    print("📊 TRANSFER LEARNING RESULTS")
    print("="*70)
    print(f"\nWith Transfer (first 100 episodes on hard students):")
    print(f"  Avg Reward: {np.mean(transfer_rewards[:100]):.1f}")
    print(f"  Avg Skill: {np.mean(transfer_skills[:100]):.3f}")
    
    print(f"\nFrom Scratch (first 100 episodes on hard students):")
    print(f"  Avg Reward: {np.mean(scratch_rewards[:100]):.1f}")
    print(f"  Avg Skill: {np.mean(scratch_skills[:100]):.3f}")
    
    improvement = ((np.mean(transfer_rewards[:100]) - np.mean(scratch_rewards[:100])) / abs(np.mean(scratch_rewards[:100]))) * 100
    print(f"\n🎯 Transfer learning improves initial performance by {improvement:.1f}%!")

if __name__ == "__main__":
    train_transfer(num_episodes_source=250, num_episodes_target=250)

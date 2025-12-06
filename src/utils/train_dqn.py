"""
Training script for DQN agent - FAST VERSION
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from environments.student_simulator import StudentSimulator
from agents.dqn_agent import DQNAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

def train_dqn(num_episodes: int = 500, save_freq: int = 100, save_dir: str = 'results/dqn'):
    os.makedirs(save_dir, exist_ok=True)
    env = StudentSimulator(max_problems=20)
    agent = DQNAgent(state_size=6, action_size=6, learning_rate=0.0005, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000, batch_size=64, target_update_freq=10)
    episode_rewards = []
    episode_lengths = []
    skill_gains = []
    accuracies = []
    engagement_scores = []
    print("🚀 Starting DQN Training (FAST MODE)...")
    print(f"Training for {num_episodes} episodes")
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            state = next_state
            episode_reward += reward
        stats = env.get_episode_stats()
        episode_rewards.append(episode_reward)
        episode_lengths.append(stats['episode_length'])
        skill_gains.append(stats['skill_gained'])
        accuracies.append(stats['accuracy'])
        engagement_scores.append(stats['final_engagement'])
        if (episode + 1) % 100 == 0:
            print(f"\n✅ Episode {episode+1}/{num_episodes}")
            print(f"  Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
            print(f"  Avg Skill: {np.mean(skill_gains[-100:]):.3f}")
            print(f"  Avg Accuracy: {np.mean(accuracies[-100:]):.2%}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            agent.save(f"{save_dir}/dqn_checkpoint_{episode+1}.pth")
    agent.save(f"{save_dir}/dqn_final.pth")
    metrics = {'episode_rewards': episode_rewards, 'episode_lengths': episode_lengths, 'skill_gains': skill_gains, 'accuracies': accuracies, 'engagement_scores': engagement_scores, 'eval_rewards': []}
    with open(f"{save_dir}/training_metrics.json", 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
    plot_training_results(metrics, save_dir)
    print(f"\n✅ Training complete! Results saved to {save_dir}")
    return agent, metrics

def plot_training_results(metrics, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DQN Training Results', fontsize=16)
    axes[0, 0].plot(metrics['episode_rewards'], alpha=0.3)
    if len(metrics['episode_rewards']) > 50:
        axes[0, 0].plot(smooth(metrics['episode_rewards'], 50), linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    axes[0, 1].plot(metrics['skill_gains'], alpha=0.3)
    if len(metrics['skill_gains']) > 50:
        axes[0, 1].plot(smooth(metrics['skill_gains'], 50), linewidth=2)
    axes[0, 1].set_title('Skill Gain')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Skill Gained')
    axes[0, 1].grid(True)
    axes[0, 2].plot(metrics['accuracies'], alpha=0.3)
    if len(metrics['accuracies']) > 50:
        axes[0, 2].plot(smooth(metrics['accuracies'], 50), linewidth=2)
    axes[0, 2].set_title('Accuracy')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].grid(True)
    axes[1, 0].plot(metrics['episode_lengths'], alpha=0.3)
    if len(metrics['episode_lengths']) > 50:
        axes[1, 0].plot(smooth(metrics['episode_lengths'], 50), linewidth=2)
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True)
    axes[1, 1].plot(metrics['engagement_scores'], alpha=0.3)
    if len(metrics['engagement_scores']) > 50:
        axes[1, 1].plot(smooth(metrics['engagement_scores'], 50), linewidth=2)
    axes[1, 1].set_title('Engagement')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Engagement')
    axes[1, 1].grid(True)
    axes[1, 2].axis('off')
    summary = f"TRAINING SUMMARY\n\nEpisodes: {len(metrics['episode_rewards'])}\n\nFinal 100:\nReward: {np.mean(metrics['episode_rewards'][-100:]):.1f}\nSkill: {np.mean(metrics['skill_gains'][-100:]):.3f}\nAccuracy: {np.mean(metrics['accuracies'][-100:]):.1%}\nEngagement: {np.mean(metrics['engagement_scores'][-100:]):.2f}"
    axes[1, 2].text(0.1, 0.5, summary, fontsize=11, family='monospace', verticalalignment='center')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_plots.png", dpi=300)
    print(f"📊 Plots saved")

def smooth(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

if __name__ == "__main__":
    train_dqn(num_episodes=500)

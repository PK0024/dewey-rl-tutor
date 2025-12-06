"""
Compare DQN and PPO agents
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from environments.student_simulator import StudentSimulator
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent

def load_metrics(method):
    """Load training metrics"""
    with open(f'results/{method}/training_metrics.json', 'r') as f:
        return json.load(f)

def compare_training_curves():
    """Compare training curves of both methods"""
    
    dqn_metrics = load_metrics('dqn')
    ppo_metrics = load_metrics('ppo')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DQN vs PPO: Training Comparison', fontsize=16, fontweight='bold')
    
    # Rewards
    axes[0, 0].plot(smooth(dqn_metrics['episode_rewards']), label='DQN', linewidth=2)
    axes[0, 0].plot(smooth(ppo_metrics['episode_rewards']), label='PPO', linewidth=2)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Skill gains
    axes[0, 1].plot(smooth(dqn_metrics['skill_gains']), label='DQN', linewidth=2)
    axes[0, 1].plot(smooth(ppo_metrics['skill_gains']), label='PPO', linewidth=2)
    axes[0, 1].set_title('Skill Gain per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Skill Gained')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(smooth(dqn_metrics['accuracies']), label='DQN', linewidth=2)
    axes[1, 0].plot(smooth(ppo_metrics['accuracies']), label='PPO', linewidth=2)
    axes[1, 0].set_title('Student Accuracy')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Engagement
    axes[1, 1].plot(smooth(dqn_metrics['engagement_scores']), label='DQN', linewidth=2)
    axes[1, 1].plot(smooth(ppo_metrics['engagement_scores']), label='PPO', linewidth=2)
    axes[1, 1].set_title('Final Engagement')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Engagement Level')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/training_comparison.png', dpi=300)
    print("✅ Training comparison saved to results/comparison/training_comparison.png")


def evaluate_both_agents(num_eval_episodes=100):
    """Evaluate both agents comprehensively"""
    
    env = StudentSimulator(max_problems=20)
    
    # Load agents
    print("Loading DQN agent...")
    dqn_agent = DQNAgent()
    dqn_agent.load('results/dqn/dqn_final.pth')
    
    print("Loading PPO agent...")
    ppo_agent = PPOAgent()
    ppo_agent.load('results/ppo/ppo_final.pth')
    
    # Baseline (random policy)
    def random_policy(state):
        return np.random.randint(0, 6)
    
    results = {
        'DQN': {'rewards': [], 'skills': [], 'accuracies': [], 'engagements': []},
        'PPO': {'rewards': [], 'skills': [], 'accuracies': [], 'engagements': []},
        'Random': {'rewards': [], 'skills': [], 'accuracies': [], 'engagements': []}
    }
    
    print(f"\n🧪 Evaluating agents ({num_eval_episodes} episodes each)...")
    
    # Evaluate DQN
    print("Evaluating DQN...")
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = dqn_agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        stats = env.get_episode_stats()
        results['DQN']['rewards'].append(episode_reward)
        results['DQN']['skills'].append(stats['skill_gained'])
        results['DQN']['accuracies'].append(stats['accuracy'])
        results['DQN']['engagements'].append(stats['final_engagement'])
    
    # Evaluate PPO
    print("Evaluating PPO...")
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _, _, _ = ppo_agent.act(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        stats = env.get_episode_stats()
        results['PPO']['rewards'].append(episode_reward)
        results['PPO']['skills'].append(stats['skill_gained'])
        results['PPO']['accuracies'].append(stats['accuracy'])
        results['PPO']['engagements'].append(stats['final_engagement'])
    
    # Evaluate Random
    print("Evaluating Random baseline...")
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = random_policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        
        stats = env.get_episode_stats()
        results['Random']['rewards'].append(episode_reward)
        results['Random']['skills'].append(stats['skill_gained'])
        results['Random']['accuracies'].append(stats['accuracy'])
        results['Random']['engagements'].append(stats['final_engagement'])
    
    # Save results
    os.makedirs('results/comparison', exist_ok=True)
    with open('results/comparison/evaluation_results.json', 'w') as f:
        json.dump({
            k: {metric: [float(v) for v in vals] for metric, vals in v.items()}
            for k, v in results.items()
        }, f)
    
    # Print statistics
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS (Mean ± Std)")
    print("="*70)
    
    for method in ['Random', 'DQN', 'PPO']:
        print(f"\n{method}:")
        print(f"  Reward:     {np.mean(results[method]['rewards']):7.2f} ± {np.std(results[method]['rewards']):5.2f}")
        print(f"  Skill Gain: {np.mean(results[method]['skills']):7.3f} ± {np.std(results[method]['skills']):5.3f}")
        print(f"  Accuracy:   {np.mean(results[method]['accuracies']):7.1%} ± {np.std(results[method]['accuracies']):5.1%}")
        print(f"  Engagement: {np.mean(results[method]['engagements']):7.2f} ± {np.std(results[method]['engagements']):5.2f}")
    
    # Calculate improvements
    print("\n" + "="*70)
    print("📈 IMPROVEMENTS OVER RANDOM BASELINE")
    print("="*70)
    
    for method in ['DQN', 'PPO']:
        print(f"\n{method} vs Random:")
        reward_improvement = (np.mean(results[method]['rewards']) - np.mean(results['Random']['rewards'])) / np.mean(results['Random']['rewards']) * 100
        skill_improvement = (np.mean(results[method]['skills']) - np.mean(results['Random']['skills'])) / np.mean(results['Random']['skills']) * 100
        accuracy_improvement = (np.mean(results[method]['accuracies']) - np.mean(results['Random']['accuracies'])) / np.mean(results['Random']['accuracies']) * 100
        engagement_improvement = (np.mean(results[method]['engagements']) - np.mean(results['Random']['engagements'])) / np.mean(results['Random']['engagements']) * 100
        
        print(f"  Reward:     +{reward_improvement:5.1f}%")
        print(f"  Skill Gain: +{skill_improvement:5.1f}%")
        print(f"  Accuracy:   +{accuracy_improvement:5.1f}%")
        print(f"  Engagement: +{engagement_improvement:5.1f}%")
    
    return results


def plot_comparison_boxplots(results):
    """Create box plots comparing methods"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: DQN vs PPO vs Random', fontsize=16, fontweight='bold')
    
    metrics = ['rewards', 'skills', 'accuracies', 'engagements']
    titles = ['Total Reward', 'Skill Gained', 'Accuracy', 'Final Engagement']
    ylabels = ['Reward', 'Skill Gain', 'Accuracy', 'Engagement']
    
    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        data = [results[method][metric] for method in ['Random', 'DQN', 'PPO']]
        bp = ax.boxplot(data, labels=['Random', 'DQN', 'PPO'], patch_artist=True)
        
        colors = ['lightgray', 'lightblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/comparison/comparison_boxplots.png', dpi=300)
    print("✅ Comparison boxplots saved to results/comparison/comparison_boxplots.png")


def smooth(data, window=50):
    """Smooth data"""
    if len(data) < window:
        return data
    return list(np.convolve(data, np.ones(window)/window, mode='valid'))


if __name__ == "__main__":
    print("="*70)
    print("DQN vs PPO Comparison Analysis")
    print("="*70)
    
    # Compare training curves
    print("\n1. Comparing training curves...")
    compare_training_curves()
    
    # Evaluate and compare
    print("\n2. Running comprehensive evaluation...")
    results = evaluate_both_agents(num_eval_episodes=100)
    
    # Create boxplots
    print("\n3. Creating comparison visualizations...")
    plot_comparison_boxplots(results)
    
    print("\n" + "="*70)
    print("✅ Comparison complete! Check results/comparison/ folder")
    print("="*70)
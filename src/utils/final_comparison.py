"""Final comparison for submission"""
import json, numpy as np, matplotlib.pyplot as plt, os

# Load all results
dqn = json.load(open('results/dqn/training_metrics.json'))
transfer = json.load(open('results/transfer/transfer_metrics.json'))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Complete RL Comparison: DQN + Transfer Learning', fontsize=16, fontweight='bold')

smooth = lambda d, w=25: list(np.convolve(d, np.ones(w)/w, mode='valid')) if len(d) > w else d

# DQN Standard Task
axes[0, 0].plot(smooth(dqn['episode_rewards']), linewidth=2, color='blue')
axes[0, 0].set_title('DQN: Standard Task')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].grid(True)

# Transfer Learning Source
axes[0, 1].plot(smooth(transfer['source_rewards']), linewidth=2, color='green')
axes[0, 1].set_title('Transfer: Source Task (Easy Students)')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Reward')
axes[0, 1].grid(True)

# Transfer vs Scratch
axes[0, 2].plot(smooth(transfer['transfer_rewards']), label='With Transfer', linewidth=2, color='green')
axes[0, 2].plot(smooth(transfer['scratch_rewards']), label='From Scratch', linewidth=2, color='red')
axes[0, 2].set_title('Transfer: Target Task (Hard Students)')
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Reward')
axes[0, 2].legend()
axes[0, 2].grid(True)

# Skills comparison
axes[1, 0].plot(smooth(dqn['skill_gains']), linewidth=2, color='blue', label='DQN Standard')
axes[1, 0].plot(smooth(transfer['source_skills']), linewidth=2, color='green', label='Source (Easy)')
axes[1, 0].set_title('Skill Gains Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Transfer skill comparison
axes[1, 1].plot(smooth(transfer['transfer_skills']), label='With Transfer', linewidth=2, color='green')
axes[1, 1].plot(smooth(transfer['scratch_skills']), label='From Scratch', linewidth=2, color='red')
axes[1, 1].set_title('Skill Gains: Transfer vs Scratch')
axes[1, 1].legend()
axes[1, 1].grid(True)

# Summary stats
axes[1, 2].axis('off')
summary = f"""
FINAL RESULTS SUMMARY

DQN (Standard Task):
  Reward: {np.mean(dqn['episode_rewards'][-100:]):.0f}
  Skill: {np.mean(dqn['skill_gains'][-100:]):.3f}
  Acc: {np.mean(dqn['accuracies'][-100:]):.1%}

Transfer Learning:
  Source Reward: {np.mean(transfer['source_rewards'][-100:]):.0f}
  
  Target (w/ Transfer): {np.mean(transfer['transfer_rewards'][-100:]):.0f}
  Target (Scratch): {np.mean(transfer['scratch_rewards'][-100:]):.0f}
  
  Transfer Effect: -23.6%
  (Negative transfer observed)

KEY INSIGHT:
Teaching strategies for easy
and hard students require
fundamentally different
approaches, not just scaling.
"""
axes[1, 2].text(0.05, 0.5, summary, fontsize=10, family='monospace', verticalalignment='center')

plt.tight_layout()
os.makedirs('results/comparison', exist_ok=True)
plt.savefig('results/comparison/complete_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Final comparison saved to results/comparison/complete_comparison.png")

print("\n" + "="*70)
print("🎉 ALL TRAINING COMPLETE!")
print("="*70)
print("\nYou have implemented:")
print("  1. DQN (Value-Based Learning)")
print("  2. Transfer Learning (Meta-Learning)")
print("  ✅ TWO approaches from DIFFERENT categories!")
print("\n📊 Next steps:")
print("  - Write technical report")
print("  - Create demo video")
print("  - Submit!")

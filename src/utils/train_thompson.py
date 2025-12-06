import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from environments.student_simulator import StudentSimulator
from agents.thompson_agent import ThompsonAgent
import json, matplotlib.pyplot as plt

env = StudentSimulator(max_problems=20)
agent = ThompsonAgent(action_size=6)
rewards, skills, accs = [], [], []

print("🚀 Thompson Sampling - 500 episodes...")
for ep in range(500):
    state, _ = env.reset()
    ep_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, term, trunc, info = env.step(action)
        done = term or trunc
        agent.update(action, reward)
        state = next_state
        ep_reward += reward
    stats = env.get_episode_stats()
    rewards.append(ep_reward)
    skills.append(stats['skill_gained'])
    accs.append(stats['accuracy'])
    if (ep + 1) % 100 == 0:
        print(f"Episode {ep+1}: Reward={np.mean(rewards[-100:]):.1f}, Skill={np.mean(skills[-100:]):.3f}")

os.makedirs('results/thompson', exist_ok=True)
agent.save('results/thompson/thompson_final.json')
json.dump({'episode_rewards': rewards, 'skill_gains': skills, 'accuracies': accs, 'episode_lengths': [], 'engagement_scores': [], 'eval_rewards': []}, 
          open('results/thompson/training_metrics.json', 'w'))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].plot(rewards)
ax[0].set_title('Rewards')
ax[1].plot(skills)
ax[1].set_title('Skills')
ax[2].plot(accs)
ax[2].set_title('Accuracy')
plt.savefig('results/thompson/training_plots.png')
print("✅ Done!")

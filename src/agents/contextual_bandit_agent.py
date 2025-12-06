"""
Contextual Bandit with UCB (Upper Confidence Bound)
"""
import numpy as np
import json

class ContextualBanditAgent:
    """UCB-based contextual bandit for action selection"""
    
    def __init__(self, state_size=6, action_size=6, exploration_param=2.0):
        self.action_size = action_size
        self.exploration_param = exploration_param
        self.action_counts = np.zeros(action_size)
        self.action_rewards = np.zeros(action_size)
        self.total_steps = 0
        
    def act(self, state, training=True):
        """Select action using UCB"""
        if training and np.min(self.action_counts) == 0:
            return np.argmin(self.action_counts)
        
        avg_rewards = self.action_rewards / (self.action_counts + 1e-8)
        exploration_bonus = self.exploration_param * np.sqrt(np.log(self.total_steps + 1) / (self.action_counts + 1))
        ucb_values = avg_rewards + exploration_bonus
        
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        """Update action statistics"""
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        self.total_steps += 1
    
    def save(self, filepath):
        """Save agent state"""
        data = {'action_counts': self.action_counts.tolist(), 'action_rewards': self.action_rewards.tolist(), 'total_steps': int(self.total_steps)}
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load agent state"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.action_counts = np.array(data['action_counts'])
        self.action_rewards = np.array(data['action_rewards'])
        self.total_steps = data['total_steps']
        print(f"Model loaded from {filepath}")

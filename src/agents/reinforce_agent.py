"""
REINFORCE Agent - FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.reinforce_network import PolicyNetwork

class REINFORCEAgent:
    """REINFORCE algorithm that actually learns"""
    
    def __init__(self, state_size=6, action_size=6, learning_rate=0.001, gamma=0.99, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.policy = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_states = []
        self.saved_actions = []
        self.rewards = []
        
    def act(self, state, training=True):
        """Select action using policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training:
            probs = self.policy(state_tensor)
            probs_np = probs.detach().cpu().numpy()[0]
            probs_np = np.clip(probs_np, 1e-8, 1.0)
            probs_np = probs_np / probs_np.sum()
            action = np.random.choice(self.action_size, p=probs_np)
            self.saved_states.append(state)
            self.saved_actions.append(action)
        else:
            with torch.no_grad():
                probs = self.policy(state_tensor)
                action = probs.argmax().item()
        
        return action
    
    def store_reward(self, reward):
        """Store reward"""
        self.rewards.append(reward)
    
    def update(self):
        """ACTUALLY update policy using gradient descent"""
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.saved_states)).to(self.device)
        actions = torch.LongTensor(self.saved_actions).to(self.device)
        
        # Forward pass
        probs = self.policy(states)
        
        # Get log probabilities for taken actions
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
        
        # Calculate loss
        loss = -(log_probs * returns).mean()
        
        # Backpropagate and update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear buffers
        self.saved_states = []
        self.saved_actions = []
        self.rewards = []
        
        return loss.item()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {filepath}")

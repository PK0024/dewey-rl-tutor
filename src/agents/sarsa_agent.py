"""
SARSA Agent (On-Policy TD Learning)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.dqn_network import DQNNetwork

class SARSAAgent:
    """SARSA: On-policy temporal difference learning"""
    
    def __init__(self, state_size=6, action_size=6, learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, device=None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA update: Use next_action (on-policy)"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        current_q = self.q_network(state_tensor)[0][action]
        
        if done:
            target_q = reward
        else:
            with torch.no_grad():
                next_q = self.q_network(next_state_tensor)[0][next_action]
            target_q = reward + self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, torch.tensor(target_q).to(self.device))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        torch.save({'q_network': self.q_network.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epsilon': self.epsilon}, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")

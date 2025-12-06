"""
Thompson Sampling - ULTRA SIMPLE
"""
import numpy as np
import json

class ThompsonAgent:
    """Thompson Sampling for action selection"""
    
    def __init__(self, action_size=6):
        self.action_size = action_size
        self.alpha = np.ones(action_size)
        self.beta = np.ones(action_size)
    
    def act(self, state, training=True):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, action, reward):
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
    
    def save(self, filepath):
        data = {'alpha': self.alpha.tolist(), 'beta': self.beta.tolist()}
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.alpha = np.array(data['alpha'])
        self.beta = np.array(data['beta'])

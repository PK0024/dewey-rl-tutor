"""
Simple REINFORCE Policy Network
"""

import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    """Simple policy network for REINFORCE"""
    
    def __init__(self, state_size: int, action_size: int):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        return self.network(state)

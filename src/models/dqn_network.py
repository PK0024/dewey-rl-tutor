"""
Deep Q-Network (DQN) neural network architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    
    Architecture:
        Input (state_size) -> 128 -> 128 -> 64 -> Output (action_size)
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes=[128, 128, 64]):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Prevent overfitting
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass: state -> Q-values"""
        return self.network(state)
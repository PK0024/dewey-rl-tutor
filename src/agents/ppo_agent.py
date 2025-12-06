"""
Proximal Policy Optimization (PPO) Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.ppo_network import ActorCriticNetwork


class PPOAgent:
    """
    PPO Agent with clipped surrogate objective
    """
    
    def __init__(
        self,
        state_size: int = 6,
        action_size: int = 6,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64,
        device: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Actor-Critic network
        self.policy = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training stats
        self.loss_history = []
        
    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float, float]:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action(
                state_tensor,
                deterministic=not training
            )
        
        return (
            action.item(),
            log_prob.item(),
            entropy.item(),
            value.item()
        )
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in rollout buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float = 0.0):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        advantage = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * advantage
            advantages.insert(0, advantage)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self, next_state: np.ndarray = None):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return 0.0
        
        # Compute final value for GAE
        if next_state is not None:
            state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value = self.policy(state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        avg_loss = total_loss / num_updates if num_updates > 0 else 0
        self.loss_history.append(avg_loss)
        
        # Clear rollout buffer
        self.clear_buffer()
        
        return avg_loss
    
    def clear_buffer(self):
        """Clear rollout buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Model loaded from {filepath}")
"""
Student Simulator Environment
Simulates a student learning math problems with varying difficulty
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any


class StudentSimulator(gym.Env):
    """
    Simulates a student learning mathematics.
    
    State Space (6 dimensions):
        - observed_skill: Noisy estimate of student's true skill (0-1)
        - progress: Fraction of session completed (0-1)
        - recent_success_rate: Success rate in last 5 problems (0-1)
        - consecutive_failures: Number of consecutive failures (0-1, normalized)
        - hint_usage_rate: Fraction of problems where hints were used (0-1)
        - engagement_level: Current engagement (0-1)
    
    Action Space (6 discrete actions):
        0: Give easy problem (difficulty 0.3)
        1: Give medium problem (difficulty 0.5)
        2: Give hard problem (difficulty 0.7)
        3: Give hint on current problem
        4: Review previous concept
        5: Advance to next topic
    
    Reward Structure:
        - Correct answer: +10 (more if appropriately challenging)
        - Incorrect answer: -5 (more if too hard)
        - Optimal challenge bonus: +5
        - Disengagement penalty: -10
        - Learning progress bonus: +skill_gained * 20
    """
    
    def __init__(self, max_problems: int = 20, difficulty_levels: int = 5):
        super().__init__()
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(6)
        
        # Episode parameters
        self.max_problems = max_problems
        self.difficulty_levels = difficulty_levels
        
        # Student parameters (will be randomized each episode)
        self.true_skill = 0.5
        self.learning_rate = 0.02  # How fast student learns
        self.frustration_threshold = 0.3  # How much failure before disengagement
        self.boredom_threshold = 0.7  # How easy before boredom
        
        # Episode state
        self.current_step = 0
        self.problems_attempted = 0
        self.problems_correct = 0
        self.consecutive_failures = 0
        self.hints_used = 0
        self.engagement = 1.0
        self.recent_results = []  # Last 5 results
        
        # For rendering/logging
        self.episode_history = []
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Randomize student characteristics
        self.true_skill = np.random.uniform(0.25, 0.75)
        self.learning_rate = np.random.uniform(0.01, 0.04)
        self.frustration_threshold = np.random.uniform(0.25, 0.35)
        self.boredom_threshold = np.random.uniform(0.65, 0.75)
        
        # Reset episode state
        self.current_step = 0
        self.problems_attempted = 0
        self.problems_correct = 0
        self.consecutive_failures = 0
        self.hints_used = 0
        self.engagement = 1.0
        self.recent_results = []
        self.episode_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Add noise to skill observation (tutor doesn't see true skill)
        observed_skill = np.clip(
            self.true_skill + np.random.normal(0, 0.05), 0, 1
        )
        
        # Calculate recent success rate
        if len(self.recent_results) > 0:
            recent_success_rate = sum(self.recent_results) / len(self.recent_results)
        else:
            recent_success_rate = 0.5  # Neutral prior
        
        # Progress through session
        progress = self.problems_attempted / self.max_problems
        
        # Normalized consecutive failures
        norm_failures = min(self.consecutive_failures / 5.0, 1.0)
        
        # Hint usage rate
        hint_rate = self.hints_used / max(self.problems_attempted, 1)
        
        state = np.array([
            observed_skill,
            progress,
            recent_success_rate,
            norm_failures,
            hint_rate,
            self.engagement
        ], dtype=np.float32)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        reward = 0.0
        info = {}
        
        # Track initial skill for learning measurement
        initial_skill = self.true_skill
        
        # Process action
        if action in [0, 1, 2]:  # Problem of varying difficulty
            difficulty = [0.3, 0.5, 0.7][action]
            solved, step_reward = self._attempt_problem(difficulty)
            reward += step_reward
            info['action_type'] = f'problem_difficulty_{difficulty}'
            info['solved'] = solved
            
        elif action == 3:  # Give hint
            reward += self._give_hint()
            info['action_type'] = 'hint'
            
        elif action == 4:  # Review
            reward += self._review_concept()
            info['action_type'] = 'review'
            
        elif action == 5:  # Next topic
            reward += self._next_topic()
            info['action_type'] = 'next_topic'
        
        # Add learning progress bonus
        skill_gained = self.true_skill - initial_skill
        reward += skill_gained * 20
        
        # Check termination conditions
        terminated = (
            self.problems_attempted >= self.max_problems or
            self.engagement < 0.2
        )
        
        truncated = False
        
        # Log this step
        self.episode_history.append({
            'step': self.current_step,
            'action': action,
            'reward': reward,
            'true_skill': self.true_skill,
            'engagement': self.engagement,
            'problems_correct': self.problems_correct,
            'problems_attempted': self.problems_attempted
        })
        
        info['true_skill'] = self.true_skill
        info['engagement'] = self.engagement
        info['total_reward'] = reward
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _attempt_problem(self, difficulty: float) -> Tuple[bool, float]:
        """Student attempts a problem"""
        self.problems_attempted += 1
        
        # Calculate success probability
        # Success = skill - difficulty + noise
        success_prob = np.clip(
            self.true_skill - difficulty + 0.25 + np.random.normal(0, 0.05),
            0.0, 1.0
        )
        
        solved = np.random.random() < success_prob
        
        # Update recent results
        self.recent_results.append(float(solved))
        if len(self.recent_results) > 5:
            self.recent_results.pop(0)
        
        # Calculate reward
        reward = 0.0
        
        if solved:
            # Base success reward
            reward += 10.0
            self.problems_correct += 1
            self.consecutive_failures = 0
            
            # Bonus for appropriate challenge (in "zone of proximal development")
            challenge_gap = abs(difficulty - self.true_skill)
            if challenge_gap < 0.15:  # Sweet spot!
                reward += 5.0
            elif challenge_gap > 0.3:  # Too easy = boring
                reward -= 3.0
                self.engagement -= 0.05
            
            # Learning occurs
            self.true_skill += self.learning_rate * (difficulty ** 1.5)
            self.true_skill = min(self.true_skill, 1.0)
            
        else:  # Failed
            reward -= 5.0
            self.consecutive_failures += 1
            
            # Strong penalty if way too hard (frustration)
            if difficulty > self.true_skill + 0.3:
                reward -= 10.0
                self.engagement -= 0.15
            else:
                # Mild failure is okay (learning opportunity)
                reward -= 2.0
                self.engagement -= 0.05
            
            # Compounding frustration
            if self.consecutive_failures >= 3:
                reward -= 5.0 * self.consecutive_failures
                self.engagement -= 0.1
        
        # Clip engagement
        self.engagement = np.clip(self.engagement, 0.0, 1.0)
        
        return solved, reward
    
    def _give_hint(self) -> float:
        """Give student a hint"""
        self.hints_used += 1
        
        # Hints help but don't substitute for learning
        reward = 2.0
        
        # Too many hints = dependency
        hint_rate = self.hints_used / max(self.problems_attempted, 1)
        if hint_rate > 0.5:
            reward -= 3.0
        
        # Slight engagement boost
        self.engagement = min(self.engagement + 0.02, 1.0)
        
        return reward
    
    def _review_concept(self) -> float:
        """Review previous concept"""
        # Review helps consolidate learning
        self.true_skill += 0.015
        self.true_skill = min(self.true_skill, 1.0)
        
        reward = 3.0
        
        # Reset frustration
        self.consecutive_failures = max(self.consecutive_failures - 1, 0)
        self.engagement = min(self.engagement + 0.05, 1.0)
        
        return reward
    
    def _next_topic(self) -> float:
        """Move to next topic"""
        # Small reward for progression
        reward = 1.0
        
        # Good if student is doing well
        if len(self.recent_results) > 0:
            recent_rate = sum(self.recent_results) / len(self.recent_results)
            if recent_rate > 0.7:
                reward += 3.0
            elif recent_rate < 0.4:  # Too early to move on!
                reward -= 5.0
        
        return reward
    
    def render(self):
        """Print current state"""
        if len(self.episode_history) > 0:
            last = self.episode_history[-1]
            print(f"Step {last['step']}: "
                  f"Skill={last['true_skill']:.2f}, "
                  f"Engagement={last['engagement']:.2f}, "
                  f"Correct={last['problems_correct']}/{last['problems_attempted']}")
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for completed episode"""
        if len(self.episode_history) == 0:
            return {}
        
        total_reward = sum(step['reward'] for step in self.episode_history)
        final_skill = self.episode_history[-1]['true_skill']
        initial_skill = self.episode_history[0]['true_skill']
        skill_gained = final_skill - initial_skill
        
        accuracy = (self.problems_correct / max(self.problems_attempted, 1))
        
        return {
            'total_reward': total_reward,
            'final_skill': final_skill,
            'skill_gained': skill_gained,
            'accuracy': accuracy,
            'problems_attempted': self.problems_attempted,
            'final_engagement': self.engagement,
            'episode_length': len(self.episode_history)
        }
"""
Test the student simulator
"""

import sys
sys.path.append('src')

from environments.student_simulator import StudentSimulator
import numpy as np

def test_environment():
    print("Testing Student Simulator Environment...")
    
    env = StudentSimulator(max_problems=10)
    
    # Test reset
    state, info = env.reset()
    print(f"\n✅ Reset successful")
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    # Test random actions
    print(f"\n🎮 Testing random actions...")
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
              f"Skill={info['true_skill']:.2f}, Engagement={info['engagement']:.2f}")
        
        if terminated or truncated:
            print(f"\n Episode ended at step {step+1}")
            break
    
    # Get episode stats
    stats = env.get_episode_stats()
    print(f"\n📊 Episode Statistics:")
    print(f"  Total Reward: {stats['total_reward']:.2f}")
    print(f"  Skill Gained: {stats['skill_gained']:.3f}")
    print(f"  Accuracy: {stats['accuracy']:.2%}")
    print(f"  Final Engagement: {stats['final_engagement']:.2f}")
    
    print(f"\n✅ All tests passed!")

if __name__ == "__main__":
    test_environment()
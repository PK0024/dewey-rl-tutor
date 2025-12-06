# RL-Driven Adaptive Tutorial Agent - Technical Report

**Author:** Pranesh Kannan Gowrishankar  
**Institution:** Northeastern University  
**Program:** Master's in Information Systems  
**Course:** Reinforcement Learning for Agentic AI Systems  
**Date:** December 2025

---

## 1. Executive Summary

This project implements an intelligent tutoring system using two reinforcement learning approaches: DQN (value-based learning) and Transfer Learning (meta-learning). DQN achieves 949 avg reward with 0.484 skill gain. Transfer learning reveals negative transfer (-23.6%), providing insights into pedagogical strategy differences across student proficiency levels.

---

## 2. Approaches Implemented

### 2.1 Deep Q-Network (DQN)
**Category:** Value-Based Learning

- Neural network: 6 → 128 → 128 → 64 → 6
- Experience replay buffer (10,000 transitions)
- Target network (updated every 10 steps)
- ε-greedy exploration (1.0 → 0.01)
- 500 training episodes

**Results:**
- Avg Reward: 949.29
- Skill Gain: 0.484 (48.4%)
- Accuracy: 58.20%

### 2.2 Transfer Learning
**Category:** Meta-Learning and Transfer Learning

**Design:**
- Source: Train on easy students (skill 0.3-0.5), 250 episodes
- Target: Transfer to hard students (skill 0.6-0.8), 250 episodes
- Comparison: Transfer vs from-scratch learning

**Results:**
- With Transfer: 1044.3 reward (first 100 episodes)
- From Scratch: 1367.3 reward (first 100 episodes)
- Transfer Effect: -23.6% (negative transfer)

---

## 3. Key Findings

**DQN Learning:**
- Successfully learns adaptive teaching strategies
- Adapts difficulty to student skill level
- Maintains student engagement effectively

**Transfer Learning Discovery:**
- Negative transfer observed between difficulty levels
- Teaching strategies for beginners vs advanced students are fundamentally different
- Suggests need for curriculum-aware architectures

---

## 4. Conclusion

This project demonstrates both the potential and limitations of RL in education. While DQN successfully learns effective teaching policies, transfer learning reveals that optimal strategies fundamentally differ across student levels rather than being scalable variants.

**Scientific Contribution:** The negative transfer finding provides valuable insights into multi-level educational system design.

---

**Author:** Pranesh Kannan Gowrishankar  
**GitHub:** [Your Link]  
**Video:** [Your Link]

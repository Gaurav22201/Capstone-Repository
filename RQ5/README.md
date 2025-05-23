# RQ5: Impact of Environment Complexity on DRL Algorithm Performance

This study explores how increasing environment complexity affects the performance of various deep reinforcement learning (DRL) algorithms, namely DQN, REINFORCE, PPO, and SAC. Complexity is defined in terms of action space type, state dimensionality, and reward density.

---

## 📌 Research Question

**How does environment complexity affect the performance of DRL algorithms?**

---

## 💡 Hypotheses

- Simpler algorithms (e.g., DQN, REINFORCE) are expected to perform well in low-complexity environments.
- Advanced algorithms (e.g., PPO, SAC) are hypothesized to scale better with increasing complexity and sparse rewards.
- SAC is expected to perform best in high-dimensional, continuous, and sparse-reward environments due to entropy-regularized exploration.

---

## 🧪 Experimental Setup

- **Environments by Complexity**:
  - Low: `CartPole-v1`, `Pendulum-v1`
  - Medium: `LunarLander-v2`, `MountainCarContinuous-v0`
  - High: `BipedalWalker-v3`, `Humanoid-v2`

- **Algorithms Tested**:
  - DQN
  - REINFORCE
  - PPO
  - SAC

- **Key Setup Details**:
  - 1 million training timesteps per environment
  - Common architecture: 2 hidden layers with ReLU activation
  - Results averaged over 5 seeds for stochastic algorithms
  - Implemented using `stable-baselines3` and `gymnasium`
  - Hardware: NVIDIA RTX 3080, 32 GB RAM

---

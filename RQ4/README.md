# RQ4: Algorithm Family Performance Across Environment Types

This study investigates how different families of deep reinforcement learning (DRL) algorithms—value-based, policy-gradient, and actor-critic—perform across varying environments characterized by discrete vs. continuous action spaces and varying levels of task complexity.

---

## 📌 Research Question

**How do algorithm families (value-based, policy-gradient, actor-critic) perform across discrete and continuous environments of varying complexity?**

---

## 💡 Hypotheses

- **DQN** (value-based) will perform well in simple, discrete environments like CartPole-v1 due to efficient Q-value estimation.
- **REINFORCE** (basic policy-gradient) will underperform in complex environments due to high gradient variance.
- **PPO** (actor-critic) is expected to generalize well across environments due to its stable policy updates.
- **A2C** (actor-critic) is hypothesized to perform reasonably well in continuous spaces but may not match PPO.
- In challenging continuous environments, actor-critic methods (especially PPO) should dominate over simpler methods.

---

## 🧪 Experimental Setup

- **Environments Tested**:
  - `CartPole-v1`: Discrete, low complexity
  - `Pendulum-v1`: Continuous, moderate complexity
  - `MountainCarContinuous-v0`: Continuous, sparse reward

- **Algorithms**:
  - **DQN** (value-based) — only for discrete tasks
  - **REINFORCE** (policy-gradient)
  - **PPO** (actor-critic)
  - **A2C** (actor-critic)

- **Evaluation Metrics**:
  - Final average reward
  - Reward variance
  - Learning stability
  - Training time

---

## ⚙️ Setup & Usage

### 🔧 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt

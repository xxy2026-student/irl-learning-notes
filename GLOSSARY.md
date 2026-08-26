# 术语对照表

读英文论文时的查询表。**遇到新术语就往下加**，这是一份会长大的清单。

只放“英文 ↔ 中文 ↔ 符号”的对应，不放解释——
解释属于 `notes/`，那里才是自己写的地方。

## 问题设定

| 英文 | 中文 |
|---|---|
| Inverse Reinforcement Learning (IRL) | 逆向强化学习 |
| Inverse Optimal Control (IOC) | 逆向最优控制（控制论那边的说法） |
| Imitation Learning | 模仿学习 |
| Behavioural Cloning (BC) | 行为复制 |
| Apprenticeship Learning | 学徒学习 |
| Demonstration / Trajectory | 示范／轨迹 |
| Expert | 专家（产生示范的那个 policy） |

## MDP 与 forward RL

| 英文 | 中文 | 符号 |
|---|---|---|
| State / Action space | 状态／动作空间 | $S$, $A$ |
| Transition dynamics | 转移动态 | $P(s' \mid s, a)$ |
| Reward function | 奖励函数 | $R(s)$ / $R(s,a,s')$ |
| Discount factor | 折扣因子 | $\gamma$ |
| Policy | 策略 | $\pi(a \mid s)$ |
| Value / Q function | 价值函数 | $V^\pi(s)$, $Q^\pi(s,a)$ |
| Occupancy measure | 占用测度 | $\rho_\pi(s,a)$ |
| State Visitation Frequency (SVF) | 状态访问频率 | $\mu(s)$ |

## 最大熵框架

| 英文 | 中文 | 符号 |
|---|---|---|
| Feature expectation | 特征期望 | $\mu(\pi)$ / $\hat\phi_E$ |
| Feature map | 特征函数 | $\phi(s)$ |
| Partition function | 配分函数 | $Z(\theta)$ |
| Boltzmann / softmax policy | 波兹曼策略 | |
| Soft value iteration | 软性值迭代 | |
| Maximum Causal Entropy | 最大因果熵 | |
| Reward shaping | 奖励塑形 | $\Phi(s)$ |

## 对抗式方法

| 英文 | 中文 |
|---|---|
| Discriminator | 判别器 |
| Generator | 生成器 |
| GAIL | Generative Adversarial Imitation Learning |
| AIRL | Adversarial Inverse Reinforcement Learning |

## 可辨识性

| 英文 | 中文 |
|---|---|
| Ill-posed | 不适定 |
| Degenerate solution | 退化解 |
| Reward ambiguity | 奖励歧义 |
| Identifiability | 可辨识性 |
| Reward hacking | 奖励钻漏洞 |

## 博弈与偏好

| 英文 | 中文 | 符号 |
|---|---|---|
| Inverse Game Theory | 逆向博弈论 | |
| Quantal Response Equilibrium (QRE) | 量化反应均衡 | $\lambda$ |
| Level-$k$ reasoning | level-$k$ 推理 | $\tau$ |
| RLHF | 人类回馈强化学习 | |
| Bradley–Terry model | Bradley–Terry 模型 | |

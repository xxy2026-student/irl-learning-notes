# Flow Matching for RL

## 方向简介
流匹配用常微分方程把噪声映射为复杂动作分布，训练比扩散更简洁，也便于一步蒸馏。本方向沿「流匹配→策略分布建模→Q/奖励引导→离线强化学习→策略梯度→在线强化学习→Boltzmann目标」展开，并显式衔接最大熵RL与IRL：轨迹分布满足p(τ)∝exp(-cost(τ))，最优策略满足π(a|s)∝exp(Q(s,a)/α)。流与扩散策略正是用来拟合这一未归一化目标，从而超越高斯策略的单峰限制，成为生成式策略的主干。高斯演员难以覆盖多峰动作，这也是本方向取代它的直接原因。

## 核心问题
没有目标分布样本时，如何训练流策略去逼近由Q定义的Boltzmann分布？能量引导、一步蒸馏与策略梯度各自如何避免对整条ODE反传？离线预训练到在线微调时，如何保持多模态表达力并维持足够探索？

## 推荐阅读顺序
核心阅读顺序：Park等Flow Q-Learning（离线流策略与一步蒸馏）→ Feng等On the Guidance of Flow Matching（能量引导）→ McAllister等Flow Matching Policy Gradients（在线策略梯度）→ Li等Reverse Flow Matching（统一Boltzmann目标）。扩展阅读依次为Shin等注入噪声的离线到在线、Agrawalla等floq批评家、Shi等后继测度表征、Yang等PolicyFlow。

## Papers
| Year | Venue | Paper | Authors | Local PDF |
| --- | --- | --- | --- | --- |
| 2025 | ICML 2025 | Flow Q-Learning（核心） | Seohong Park, Qiyang Li, Sergey Levine | [2025_Park_Flow_Q_Learning.pdf](./2025_Park_Flow_Q_Learning.pdf) |
| 2025 | ICML 2025 | On the Guidance of Flow Matching（核心） | Ruiqi Feng, Chenglei Yu, Wenhao Deng, Peiyan Hu, Tailin Wu | [2025_Feng_Guidance_Flow_Matching.pdf](./2025_Feng_Guidance_Flow_Matching.pdf) |
| 2026 | ICLR 2026 | Flow Matching Policy Gradients（核心） | David McAllister, Songwei Ge, Brent Yi, Chung Min Kim, Ethan Weber, Hongsuk Choi, Haiwen Feng, Angjoo Kanazawa | [2026_McAllister_Flow_Matching_Policy_Gradients.pdf](./2026_McAllister_Flow_Matching_Policy_Gradients.pdf) |
| 2026 | ICML 2026 | Reverse Flow Matching: A Unified Framework for Online Reinforcement Learning with Diffusion and Flow Policies（核心） | Zeyang Li, Sunbochen Tang, Navid Azizan | [2026_Li_Reverse_Flow_Matching.pdf](./2026_Li_Reverse_Flow_Matching.pdf) |
| 2026 | ICLR 2026 | Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning（扩展） | Yongjae Shin, Jongseong Chae, Jongeui Park, Youngchul Sung | [2026_Shin_FM_Injected_Noise.pdf](./2026_Shin_FM_Injected_Noise.pdf) |
| 2026 | ICLR 2026 | floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL（扩展） | Bhavya Agrawalla, Michal Nauman, Khush Agrawal, Aviral Kumar | [2026_Agrawalla_floq.pdf](./2026_Agrawalla_floq.pdf) |
| 2026 | ICLR 2026 | Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations（扩展） | Haosen Shi, Jianda Chen, Sinno Jialin Pan | [2026_Shi_Successor_Flow_Representations.pdf](./2026_Shi_Successor_Flow_Representations.pdf) |
| 2026 | ICLR 2026 | PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning（扩展） | Shunpeng Yang, Ben Liu, Hua Chen | [2026_Yang_PolicyFlow.pdf](./2026_Yang_PolicyFlow.pdf) |

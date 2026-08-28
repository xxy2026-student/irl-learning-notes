# Generative Policies for RL

## 方向简介
传统高斯策略默认单峰、对角协方差，似然易算，却难以表达多模态动作与复杂操作技能。扩散模型把策略写成条件生成过程，用迭代去噪覆盖多峰动作分布，并以Q函数或奖励引导采样，从而在离线与在线强化学习中取代高斯演员。本方向覆盖轨迹级扩散规划、离线扩散策略、基于Q-score的策略更新，以及无需从最优策略采样即可在线训练扩散策略，解释高斯策略为何被扩散与流模型取代。

## 核心问题
如何把生成模型当作策略类，同时保持多模态表达力与可训练性？Q函数或奖励如何注入反向扩散，既提升回报又不破坏行为克隆正则？如何避免对整条去噪链反传，把离线学到的扩散策略高效接到在线强化学习？

## 推荐阅读顺序
先读Janner等Diffuser，理解轨迹扩散与价值引导规划；再读Wang等Diffusion-QL，掌握离线设定下条件扩散策略与Q引导；接着读Psenka等Q-score matching，把策略score与动作梯度∇_a Q对齐；最后读Ma等在线扩散策略，用重加权score matching完成离线到在线的衔接。

## Papers
| Year | Venue | Paper | Authors | Local PDF |
| --- | --- | --- | --- | --- |
| 2022 | ICML 2022 | Planning with Diffusion for Flexible Behavior Synthesis | Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine | [2022_Janner_Diffuser.pdf](./2022_Janner_Diffuser.pdf) |
| 2023 | ICLR 2023 | Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning | Zhendong Wang, Jonathan J. Hunt, Mingyuan Zhou | [2023_Wang_Diffusion_Policies_Offline_RL.pdf](./2023_Wang_Diffusion_Policies_Offline_RL.pdf) |
| 2024 | ICML 2024 | Learning a Diffusion Model Policy from Rewards via Q-Score Matching | Michael Psenka, Alejandro Escontrela, Pieter Abbeel, Yi Ma | [2024_Psenka_Q_Score_Matching.pdf](./2024_Psenka_Q_Score_Matching.pdf) |
| 2025 | ICML 2025 | Efficient Online Reinforcement Learning for Diffusion Policy | Haitong Ma, Tianyi Chen, Kai Wang, Na Li, Bo Dai | [2025_Ma_Online_Diffusion_Policy.pdf](./2025_Ma_Online_Diffusion_Policy.pdf) |

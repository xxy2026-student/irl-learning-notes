# Latent与Switching Reward IRL

## 方向简介
标准逆强化学习假设单一、平稳且马尔可夫的奖励，难以解释动物与机器人在未知潜在情境、切换环境与混合目标下的长期行为。本方向关注潜在上下文、切换奖励、奖励混合、历史依赖奖励与潜在任务识别，用隐变量或切换过程把无标注示范分解为多个可解释的奖励模态，而不是假设全局共用一个奖励。

## 核心问题
如何从无标注轨迹推断离散意图切换，并同时学习历史依赖奖励？潜在上下文应建模为记忆缺失的马尔可夫链、固定窗口状态增广，还是循环网络？如何把可观测协变量与真正隐变量分开，避免把情境误当成潜在任务？

## 推荐阅读顺序
先读Ke等SWIRL，掌握切换奖励与历史依赖；再读Zhu等HIQL/多意图IQL，理解离散意图分段；最后读Sheng等PRISM，看循环意图门控如何替代固定窗口。对2023–2026年arXiv、OpenReview与PMLR的检索只保留高度相关文献，未填充泛化多任务强化学习。

## Papers
| Year | Venue | Paper | Authors | Local PDF |
| --- | --- | --- | --- | --- |
| 2024 | TMLR 2024 | Multi-intention Inverse Q-learning for Interpretable Behavior Representation | Hao Zhu, Brice De La Crompe, Gabriel Kalweit, Artur Schneider, Maria Kalweit, Ilka Diester, Joschka Boedecker | [2024_Zhu_Multi_Intention_IQL.pdf](./2024_Zhu_Multi_Intention_IQL.pdf) |
| 2025 | ICML 2025 | Inverse Reinforcement Learning with Switching Rewards and History Dependency for Characterizing Animal Behaviors | Jingyang Ke, Feiyang Wu, Jiyi Wang, Jeffrey Markowitz, Anqi Wu | [2025_Ke_Switching_Rewards_IRL.pdf](./2025_Ke_Switching_Rewards_IRL.pdf) |
| 2026 | arXiv / Preprint | Probabilistic Recurrent Intention Switching Model | Wenyuan Sheng, Hao Zhu, Joschka Boedecker | [2026_Sheng_PRISM.pdf](./2026_Sheng_PRISM.pdf) |

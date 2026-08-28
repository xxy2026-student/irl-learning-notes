# Offline IRL与Reward Identifiability

## 方向简介
离线逆强化学习仅能从预先采集的示范轨迹推断奖励，不能主动与环境或专家交互。IRL固有病态：无数奖励可解释同一行为，因而研究重心从点估计转向可行奖励集与奖励等价类。本方向在数据覆盖不受控的离线设定下，刻画可辨识性、包含单调性与样本复杂度，并比较IRL相对标准RL的统计难度，同时把辨识对象从奖励推广到风险敏感效用。

## 核心问题
核心在于：离线示范下可行奖励集如何定义与一致估计？奖励等价如何量化，悲观原则能否保证估计集包含真实可行集？IRL是否比标准强化学习更难，多项式样本与多项式时间是否足够？大状态空间或线性MDP中，可行集是否仍可有效学习，抑或必须改用奖励相容性与分类式框架？示范能否揭示风险态度，而非仅恢复风险中性奖励？

## 推荐阅读顺序
建议先读Zhao等，比较IRL与离线、在线RL的样本复杂度，理解悲观奖励学习与迁移保证；再读Lazzati等提出的离线可行集以及IRLO与PIRLO算法；接着读大状态空间上的奖励相容性与CATY-IRL，看清可行集可扩展性的边界；最后读效用学习，理解多环境示范如何缓解效用的部分可辨识性。

## Papers
| Year | Venue | Paper | Authors | Local PDF |
| --- | --- | --- | --- | --- |
| 2024 | ICML 2024 | Is Inverse Reinforcement Learning Harder than Standard Reinforcement Learning? A Theoretical Perspective | Lei Zhao, Mengdi Wang, Yu Bai | [2024_Zhao_IRL_Harder_than_RL.pdf](./2024_Zhao_IRL_Harder_than_RL.pdf) |
| 2024 | ICML 2024 | Offline Inverse RL: New Solution Concepts and Provably Efficient Algorithms | Filippo Lazzati, Mirco Mutti, Alberto Maria Metelli | [2024_Lazzati_Offline_IRL.pdf](./2024_Lazzati_Offline_IRL.pdf) |
| 2024 | NeurIPS 2024 | How does Inverse RL Scale to Large State Spaces? A Provably Efficient Approach | Filippo Lazzati, Mirco Mutti, Alberto Maria Metelli | [2024_Lazzati_Large_State_Spaces.pdf](./2024_Lazzati_Large_State_Spaces.pdf) |
| 2025 | ICML 2025 | Learning Utilities from Demonstrations in Markov Decision Processes | Filippo Lazzati, Alberto Maria Metelli | [2025_Lazzati_Learning_Utilities.pdf](./2025_Lazzati_Learning_Utilities.pdf) |

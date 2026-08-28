# Active IRL与Environment Design

## 方向简介
被动观察示范往往覆盖不足，难以把可行奖励集收缩到可迁移的程度。主动IRL把主动学习与探测引入奖励推断：学习者选择查询状态、轨迹或探测策略，以尽快识别专家奖励。更进一步，环境与输入设计把实验设计思想带入IRL——由学习者决定专家所处的环境或任务，从而同时提高样本效率与跨动力学稳健性。本方向沿被动观察、主动干预、环境设计到更可辨识奖励这条主线展开，并覆盖信息论主动探测。

## 核心问题
核心问题包括：在没有生成模型、只能顺序交互时，如何主动探索以识别奖励并找到好策略？应当查询单个动作标注、整条轨迹，还是设计全新环境？如何用贝叶斯实验设计或后悔准则选择环境以最大化信息增益？主动探测怎样为学徒策略提供PAC式可靠保证，而不只是启发式查询？

## 推荐阅读顺序
建议先读Metelli等在基础文献中的可迁移奖励与主动采样TRAVEL，建立可行集与迁移误差的理论图景；再读Lindner等AceIRL，掌握无生成模型下的主动探索与样本复杂度；然后读Kleine Buening等的环境设计框架ED-BIRL与ED-AIRL；可选读Bajgar等基于期望信息增益的贝叶斯主动IRL，理解面向PAC学徒学习的查询策略。

## Papers
| Year | Venue | Paper | Authors | Local PDF |
| --- | --- | --- | --- | --- |
| 2021 | ICML 2021 | Provably Efficient Learning of Transferable Rewards | Alberto Maria Metelli, Giorgia Ramponi, Alessandro Concetti, Marcello Restelli | [2021_Metelli_Transferable_Rewards.pdf](../../基础文献/2021_Metelli_Transferable_Rewards.pdf) （全文在基础文献，本目录不重复存放） |
| 2022 | NeurIPS 2022 | Active Exploration for Inverse Reinforcement Learning | David Lindner, Andreas Krause, Giorgia Ramponi | [2022_Lindner_Active_Exploration_IRL.pdf](./2022_Lindner_Active_Exploration_IRL.pdf) |
| 2024 | ICML 2024 | Environment Design for Inverse Reinforcement Learning | Thomas Kleine Buening, Victor Villin, Christos Dimitrakakis | [2024_KleineBuening_Environment_Design_IRL.pdf](./2024_KleineBuening_Environment_Design_IRL.pdf) |
| 2025 | RLC 2025 | PAC Apprenticeship Learning with Bayesian Active Inverse Reinforcement Learning | Ondrej Bajgar, Dewi S. W. Gould, Jonathon Liu, Alessandro Abate, Konstantinos Gatsis, Michael A. Osborne | [2025_Bajgar_PAC_Active_IRL.pdf](./2025_Bajgar_PAC_Active_IRL.pdf) |

# Multi-Agent IRL与Inverse Games

## 方向简介
多智能体场景中，观测到的是均衡行为而非单智能体最优策略。研究脉络从单智能体IRL走向策略互动，再进入逆博弈：根据Nash、Stackelberg或有限理性（如量化响应）观测反推未知效用。均衡多重性使可行奖励集更加含糊，甚至可能改变博弈的合作或竞争属性；熵正则与有限理性带来的光滑性，往往能改善可辨识性并给出更紧的样本界。本方向强调均衡观测下的奖励与效用识别，以及多智能体可行集的刻画。

## 核心问题
核心问题：如何从Stackelberg或Nash均衡观测识别效用参数？把完美理性放松为有限理性，究竟是统计上的祝福还是诅咒？在竞争矩阵博弈与马尔可夫博弈中，熵正则量化响应均衡能否唯一化均衡并构造奖励置信集？多智能体可行奖励集如何定义，其样本复杂度相对直接学习均衡有何本质增加？

## 推荐阅读顺序
建议先读Wu等，理解有限理性下Stackelberg逆博弈为何可高效学习；再读Goktas等将逆多智能体学习写成可多项式时间求解的生成对抗极小极大问题；然后读Liao等在竞争博弈中解码奖励并给出可辨识条件；最后读Freihaut等，系统刻画多智能体可行奖励集与熵正则下的误差传播。

## Papers
| Year | Venue | Paper | Authors | Local PDF |
| --- | --- | --- | --- | --- |
| 2022 | NeurIPS 2022 | Inverse Game Theory for Stackelberg Games: the Blessing of Bounded Rationality | Jibang Wu, Weiran Shen, Fei Fang, Haifeng Xu | [2022_Wu_Inverse_Game_Theory_Stackelberg.pdf](./2022_Wu_Inverse_Game_Theory_Stackelberg.pdf) |
| 2024 | ICLR 2024 | Efficient Inverse Multiagent Learning | Denizalp Goktas, Amy Greenwald, Sadie Zhao, Alec Koppel, Sumitra Ganesh | [2024_Goktas_Efficient_Inverse_Multiagent.pdf](./2024_Goktas_Efficient_Inverse_Multiagent.pdf) |
| 2025 | ICML 2025 | Decoding Rewards in Competitive Games: Inverse Game Theory with Entropy Regularization | Junyi Liao, Zihan Zhu, Ethan X. Fang, Zhuoran Yang, Vahid Tarokh | [2025_Liao_Decoding_Rewards_Inverse_Games.pdf](./2025_Liao_Decoding_Rewards_Inverse_Games.pdf) |
| 2025 | NeurIPS 2025 | On Feasible Rewards in Multi-Agent Inverse Reinforcement Learning | Till Freihaut, Giorgia Ramponi | [2025_Freihaut_Feasible_Rewards_MAIRL.pdf](./2025_Freihaut_Feasible_Rewards_MAIRL.pdf) |

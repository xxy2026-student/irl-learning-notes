# IRL 基础文献

阅读主线：**Classical IRL → Feature Matching → Maximum Entropy IRL → Maximum Causal Entropy → Deep / Adversarial IRL → Reward Identifiability → Modern IRL Theory**

本目录为纸质阅读库的基础半边：11 篇 canonical 论文的本地 PDF。优先收录会议 camera-ready（PMLR / NeurIPS / AAAI / 作者页）；AIRL 的 OpenReview 直链返回 403，改用 arXiv v2（文内标明 ICLR 2018）。

## 推荐阅读顺序

1. [2000_Ng_Algorithms_for_IRL.pdf](2000_Ng_Algorithms_for_IRL.pdf) — 经典 IRL：从演示反推奖励与可行奖励集合
2. [2004_Abbeel_Apprenticeship_Learning.pdf](2004_Abbeel_Apprenticeship_Learning.pdf) — 特征匹配 / 学徒学习：匹配专家特征期望即可模仿
3. [2008_Ziebart_MaxEnt_IRL.pdf](2008_Ziebart_MaxEnt_IRL.pdf) — 最大熵 IRL：用轨迹分布消歧
4. [2010_Ziebart_Max_Causal_Entropy.pdf](2010_Ziebart_Max_Causal_Entropy.pdf) — 最大因果熵：序贯决策与反馈
5. [2016_Finn_Guided_Cost_Learning.pdf](2016_Finn_Guided_Cost_Learning.pdf) — 深度 IOC：样本近似配分函数 + 策略优化
6. [2016_Ho_GAIL.pdf](2016_Ho_GAIL.pdf) — 对抗模仿：占用测度匹配 / GAN
7. [2018_Fu_AIRL.pdf](2018_Fu_AIRL.pdf) — 对抗 IRL：可迁移、解耦的奖励
8. [2021_Garg_IQ_Learn.pdf](2021_Garg_IQ_Learn.pdf) — 现代 IRL/IL：用单个 Q 隐式表示奖励与策略
9. [2021_Kim_Reward_Identification.pdf](2021_Kim_Reward_Identification.pdf) — 奖励可辨识性：何时能从最优行为还原奖励
10. [2021_Metelli_Transferable_Rewards.pdf](2021_Metelli_Transferable_Rewards.pdf) — 可行奖励集与可迁移奖励的样本复杂度
11. [2023_Metelli_Theoretical_Understanding_IRL.pdf](2023_Metelli_Theoretical_Understanding_IRL.pdf) — 现代 IRL 理论：可行奖励集的 PAC / minimax 下界

## Papers

| Year | Venue | Paper | Authors | Local PDF |
|------|-------|-------|---------|-----------|
| 2000 | ICML | Algorithms for Inverse Reinforcement Learning | Andrew Y. Ng, Stuart Russell | [2000_Ng_Algorithms_for_IRL.pdf](2000_Ng_Algorithms_for_IRL.pdf) |
| 2004 | ICML | Apprenticeship Learning via Inverse Reinforcement Learning | Pieter Abbeel, Andrew Y. Ng | [2004_Abbeel_Apprenticeship_Learning.pdf](2004_Abbeel_Apprenticeship_Learning.pdf) |
| 2008 | AAAI | Maximum Entropy Inverse Reinforcement Learning | Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, Anind K. Dey | [2008_Ziebart_MaxEnt_IRL.pdf](2008_Ziebart_MaxEnt_IRL.pdf) |
| 2010 | ICML | Modeling Interaction via the Principle of Maximum Causal Entropy | Brian D. Ziebart, J. Andrew Bagnell, Anind K. Dey | [2010_Ziebart_Max_Causal_Entropy.pdf](2010_Ziebart_Max_Causal_Entropy.pdf) |
| 2016 | ICML | Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization | Chelsea Finn, Sergey Levine, Pieter Abbeel | [2016_Finn_Guided_Cost_Learning.pdf](2016_Finn_Guided_Cost_Learning.pdf) |
| 2016 | NeurIPS | Generative Adversarial Imitation Learning | Jonathan Ho, Stefano Ermon | [2016_Ho_GAIL.pdf](2016_Ho_GAIL.pdf) |
| 2018 | ICLR | Learning Robust Rewards with Adversarial Inverse Reinforcement Learning | Justin Fu, Katie Luo, Sergey Levine | [2018_Fu_AIRL.pdf](2018_Fu_AIRL.pdf) |
| 2021 | NeurIPS | IQ-Learn: Inverse soft-Q Learning for Imitation | Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, Stefano Ermon | [2021_Garg_IQ_Learn.pdf](2021_Garg_IQ_Learn.pdf) |
| 2021 | ICML | Reward Identification in Inverse Reinforcement Learning | Kuno Kim, Shivam Garg, Kirankumar Shiragur, Stefano Ermon | [2021_Kim_Reward_Identification.pdf](2021_Kim_Reward_Identification.pdf) |
| 2021 | ICML | Provably Efficient Learning of Transferable Rewards | Alberto Maria Metelli, Giorgia Ramponi, Alessandro Concetti, Marcello Restelli | [2021_Metelli_Transferable_Rewards.pdf](2021_Metelli_Transferable_Rewards.pdf) |
| 2023 | ICML | Towards Theoretical Understanding of Inverse Reinforcement Learning | Alberto Maria Metelli, Filippo Lazzati, Marcello Restelli | [2023_Metelli_Theoretical_Understanding_IRL.pdf](2023_Metelli_Theoretical_Understanding_IRL.pdf) |

## 核验更正

- **AIRL**：官方 venue 为 **ICLR 2018**（不是 ICML 2018）。OpenReview `rkHywl-A-`，arXiv:1710.11248。
- **Kim et al. 2021**：官方标题 *Reward Identification in Inverse Reinforcement Learning*；作者 Kuno Kim, Shivam Garg, Kirankumar Shiragur, Stefano Ermon；ICML 2021，PMLR 139:5496–5505。
- **Finn 2016**：官方全名为 *Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization*（PMLR v48）。本地文件复用已核验的 arXiv v3（带 ICML 2016 页眉）。

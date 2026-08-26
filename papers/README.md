# 阅读清单

依“该用什么顺序读”排序，不是依年份。
状态栏自己维护：`⬜ 未读` → `📖 读了一半` → `✅ 已精读`（有笔记才算）。

“主题”栏只写这篇在处理什么问题，不写结论——结论读完自己写进 `notes/`。

**已读 5 篇**（B 组 2 篇、C 组 2 篇、D 组 1 篇），整理成一条线写在
[`notes/README.md`](../notes/README.md)。

2000 年代的论文多半早于 arXiv，只给会议出处；有 arXiv 编号的直接标出来。

---

## A. 前置：forward RL

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ⬜ | Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) | 书，Ch. 3–4, 13 | MDP / Bellman / policy gradient 的共同语言 |
| ⬜ | Ng, Harada & Russell, *Policy Invariance Under Reward Transformations* | ICML 1999 | 什么样的 reward 改动不会改变最优策略。**先读这篇，阶段 4 会一直回来引用** |
| ⬜ | Haarnoja et al., *Reinforcement Learning with Deep Energy-Based Policies* | ICML 2017 ([arXiv:1702.08165](https://arxiv.org/abs/1702.08165)) | soft Q-learning；最大熵框架的 forward 版本 |

## B. 经典 IRL

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ✅ | Ng & Russell, *Algorithms for Inverse Reinforcement Learning* | [ICML 2000](https://dl.acm.org/doi/10.5555/645529.657801) | IRL 的问题定义，以及它为什么不适定 |
| ✅ | Abbeel & Ng, *Apprenticeship Learning via Inverse Reinforcement Learning* | [ICML 2004](http://ai.stanford.edu/~pabbeel/irl/) | 只用轨迹（不需要完整 policy），目标从“推 reward”改成“表现一样好” |
| ⬜ | Ratliff, Bagnell & Zinkevich, *Maximum Margin Planning* | ICML 2006 | 把 IRL 当成 structured prediction |
| ⬜ | Ramachandran & Amir, *Bayesian Inverse Reinforcement Learning* | IJCAI 2007 | 把 reward 当成后验分布而非点估计 |

## C. 最大熵框架

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ✅ | Ziebart, Maas, Bagnell & Dey, *Maximum Entropy IRL* | [AAAI 2008](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf) | 用最大熵原理处理 IRL 的多解问题。**阶段 2 要自己实现的就是这篇** |
| ✅ | Ziebart, Bagnell & Dey, *Modeling Interaction via the Principle of Maximum Causal Entropy* | [ICML 2010](https://dl.acm.org/doi/10.5555/3104322.3104481) | 随机 dynamics 下的正确版本 |
| ⬜ | Wulfmeier, Ondruska & Posner, *Maximum Entropy Deep IRL* | arXiv:1507.04888 | 把线性 reward 换成神经网络 |

## D. 对抗式 / 可扩展 IRL

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ✅ | Finn, Levine & Abbeel, *Guided Cost Learning* | [ICML 2016](https://proceedings.mlr.press/v48/finn16.html) · [arXiv:1603.00448](https://arxiv.org/abs/1603.00448) | 用采样近似 partition function，脱离 tabular |
| ⬜ | Ho & Ermon, *Generative Adversarial Imitation Learning* | NeurIPS 2016 ([arXiv:1606.03476](https://arxiv.org/abs/1606.03476)) | occupancy measure matching 与 GAN 的关系 |
| ⬜ | Fu, Luo & Levine, *Learning Robust Rewards with Adversarial IRL* | ICLR 2018 ([arXiv:1710.11248](https://arxiv.org/abs/1710.11248)) | 从 discriminator 里取出可转移的 reward |

## E. 可辨识性与失效模式

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ⬜ | Amin & Singh, *Towards Resolving Unidentifiability in IRL* | arXiv:1601.06569 | 不可辨识性的分类，以及怎么缩小它 |
| ⬜ | Cao, Cohen & Szpruch, *Identifiability in Inverse Reinforcement Learning* | NeurIPS 2021 | reward 可唯一决定的充分条件 |
| ⬜ | Skalse, Howe, Krasheninnikov & Krueger, *Defining and Characterizing Reward Hacking* | NeurIPS 2022 | 代理 reward 何时会被钻漏洞 |
| ⬜ | Arora & Doshi, *A Survey of Inverse Reinforcement Learning* | Artificial Intelligence 297 ([arXiv:1806.06877](https://arxiv.org/abs/1806.06877)) | 综述。卡住的时候回来翻 |

## F. 多方互动：IRL × 博弈论（接 `game-theory-for-LLM`）

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ⬜ | Waugh, Ziebart & Bagnell, *Computational Rationalization: The Inverse Equilibrium Problem* | ICML 2011 | 最大熵框架的多方推广。**两个仓库之间最直接的桥** |
| ⬜ | Hadfield-Menell, Dragan, Abbeel & Russell, *Cooperative IRL* | NeurIPS 2016 ([arXiv:1606.03137](https://arxiv.org/abs/1606.03137)) | 把 IRL 写成人机合作博弈 |
| ⬜ | Yu, Song & Ermon, *Multi-Agent Adversarial IRL* | ICML 2019 | AIRL 推广到 Markov games |
| ⬜ | Kuleshov & Schrijvers, *Inverse Game Theory: Learning Utilities in Succinct Games* | WINE 2015 | 直接从均衡反推 utility |

## G. 偏好式 IRL = RLHF

| 状态 | 论文 | 出处 | 主题 |
|---|---|---|---|
| ⬜ | Christiano et al., *Deep RL from Human Preferences* | NeurIPS 2017 ([arXiv:1706.03741](https://arxiv.org/abs/1706.03741)) | 把示范换成成对比较 |
| ⬜ | Rafailov et al., *Direct Preference Optimization* | NeurIPS 2023 ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) | 跳过 reward model 直接优化 policy |

---

## 读法

每篇精读走同一套流程，写进 `notes/`：

1. **只读 abstract + intro 的最后一段**，先用一句话写下它在解什么问题。
2. 找到**主要的那个式子**，确认每个符号的定义域与 shape。
3. 问自己：**这个方法在什么情况下会给出错误答案？** 写下具体反例。
4. 能实现的就实现最小版本；不能实现的，写下“为什么在这个规模下做不了”。

第 3 步最重要。读完一篇却写不出反例，通常代表只是把 abstract 换句话说了一遍。

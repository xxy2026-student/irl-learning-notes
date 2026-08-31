# IRL与LLM对齐

## 方向简介
大语言模型的 post-training 是 IRL 思想近年最大的应用场：语言任务没有天然奖励，helpfulness、无害性、风格都必须从人类数据（示范、成对偏好、AI 反馈）中构造。SFT 即行为克隆，RLHF 即 Bradley-Terry 偏好反推奖励加 KL 正则的 soft RL，DPO 即利用奖励与最优策略的双射把两步并作一步——与 IQ-Learn 对 MaxEnt-IRL 的代换同构。LLM 的 MDP 高度退化（转移已知且确定、终端稀疏奖励、强参考先验），经典 IRL 的部分困难消失，而不可辨识性与分布偏移则以 reward hacking / 过优化的形态复发。

## 核心问题
核心在于：偏好数据的不可辨识方向（逐 prompt 平移）何时恰好是策略优化不需要的方向，process reward 为何破坏这一豁免？奖励模型只在标注分布上受约束，被策略主动搜索薄弱区域时，KL 约束、早停与 ensemble 之外是否存在有理论保证的悲观方案？隐式奖励（DPO）换取单层优化后失去的可迁移性与 OOD 行为如何量化？偏好非传递、标注者异质时，标量奖励假设本身失效，博弈论解概念（偏好博弈的纳什均衡）能否替代 Bradley-Terry？

## 推荐阅读顺序
先读综述（Sun & van der Schaar）建立全景与统一记号；然后按历史线补三篇原始文献：Christiano 等的偏好 RL 起点、InstructGPT 的完整流水线、DPO 的代换推导，读 DPO 时对照基础文献中的 IQ-Learn。之后读 Gao 等的过优化 scaling law，与 01_Offline_IRL 的悲观原则对照。最后两篇通往前沿：From r to Q* 处理 token 级 credit assignment，Nash-LHF 抛弃 Bradley-Terry 走向博弈论解概念，后者与 03_Multi-Agent_IRL 及 game-theory-for-LLM 的工作衔接。

## Papers

PDF 待下载（本目录建立时所在环境无法访问 arXiv），下载后把链接列改成本地文件名。

| Year | Venue | Paper | Authors | 链接 |
| --- | --- | --- | --- | --- |
| 2025 | arXiv | Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities | Hao Sun, Mihaela van der Schaar | [arXiv:2507.13158](https://arxiv.org/abs/2507.13158) |
| 2017 | NeurIPS 2017 | Deep Reinforcement Learning from Human Preferences | Paul Christiano et al. | [arXiv:1706.03741](https://arxiv.org/abs/1706.03741) |
| 2022 | NeurIPS 2022 | Training Language Models to Follow Instructions with Human Feedback | Long Ouyang et al. | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| 2023 | NeurIPS 2023 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | Rafael Rafailov et al. | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| 2023 | ICML 2023 | Scaling Laws for Reward Model Overoptimization | Leo Gao, John Schulman, Jacob Hilton | [arXiv:2210.10760](https://arxiv.org/abs/2210.10760) |
| 2024 | arXiv | From r to Q\*: Your Language Model is Secretly a Q-Function | Rafael Rafailov et al. | [arXiv:2404.12358](https://arxiv.org/abs/2404.12358) |
| 2024 | ICML 2024 | Nash Learning from Human Feedback | Rémi Munos et al. | [arXiv:2312.00886](https://arxiv.org/abs/2312.00886) |

配套：ACL 2025 tutorial "Inverse Reinforcement Learning Meets Large Language Model Alignment"（[ACL Anthology](https://aclanthology.org/2025.acl-tutorials.1/) · [slides](https://sites.google.com/view/irl-llm)）。精读导读见 [`notes/2025_Sun_IRL_meets_LLM_Survey.md`](../../../notes/2025_Sun_IRL_meets_LLM_Survey.md)。

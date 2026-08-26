# 学习路线

四个阶段。每一格的 checkbox 只有在**做得出来**时才勾——
代码跑得出正确结果，或笔记里写得出具体反例。“看过了”不算。

各阶段的细目写成**要回答的问题**，不是要记住的结论。
读之前先看一眼问题，读的时候就知道要找什么。

> 目前打勾的都是**读完并写进 `notes/README.md`** 的项目。
> 所有“自己实现”“动手”的项目一律还没动——那些要代码跑得出来才算。

---

## 阶段 0：前置（forward RL）

没有这些，IRL 论文的每个式子都会卡住。
而且每个 IRL 算法的内循环都是一次 forward RL——这里不扎实，后面 debug 会很痛苦。

- [x] MDP 五元组 $(S, A, P, R, \gamma)$、Bellman 方程
- [ ] value iteration：自己实现，并想一个独立的方法交叉验证 $V^*$ 算对了
- [x] soft value iteration：$\max$ 换成 $\log\sum\exp$ 之后，policy 变成什么？为什么 IRL 要用它？
- [ ] state visitation frequency (SVF) 的前向递推：$\mu$ 的物理意义是什么？有没有守恒量可以拿来检查？
- [ ] policy gradient / REINFORCE（阶段 3 才会需要，可以先跳过）

**验收**：`code/` 里跑得起来的 MDP solver + 测试。
**笔记**：`notes/00-...`

---

## 阶段 1：经典 IRL（线性 reward）

- [x] **Ng & Russell (2000)**
  - IRL 的问题设定长什么样？输入是什么、输出是什么？
  - 为什么这个问题是 ill-posed 的？最明显的退化解是什么？
  - 论文加了哪些额外准则来挑一个解？那些准则是推导出来的还是选择？
- [x] **Abbeel & Ng (2004)**
  - “feature expectation matching”在 match 什么？为什么不直接 match policy？
  - 为什么算法返回的是混合 policy 而不是单一 policy？
- [ ] Ratliff et al. (2006) Maximum Margin Planning（选读）
- [ ] Ramachandran & Amir (2007) Bayesian IRL（选读）

**验收**：能说清楚 Ng & Russell 留下的两个缺口，以及后面各篇分别在补哪一个。
**笔记**：`notes/01-...`、`notes/02-...`

---

## 阶段 2：最大熵框架

- [x] **Ziebart et al. (2008) MaxEnt IRL**
  - “最大熵”在这里解决的是哪个问题？
  - 梯度长什么样？为什么算它需要解一次 MDP？
- [ ] **自己实现出来**，并在已知真值的环境上验证（reward 藏起来 → 示范 → 反推 → 比对）
- [ ] 动手实验：把 expert 从 Boltzmann 换成 hard-max，反推结果会怎么变？先写下预测再跑
- [x] Ziebart et al. (2010) Maximum **Causal** Entropy：随机 dynamics 下 2008 版差在哪
- [ ] Wulfmeier et al. (2015)：reward 换成神经网络，梯度形式要不要改

**验收**：自己写的 MaxEnt IRL 能在 GridWorld 上还原出 reward。
**笔记**：`notes/03-...`

---

## 阶段 3：对抗式 / 深度 IRL

- [x] Finn et al. (2016) Guided Cost Learning：tabular 方法卡在哪一步，这篇怎么绕开
- [ ] Ho & Ermon (2016) GAIL：occupancy measure 是什么？为什么会变成 GAN？
- [ ] Fu et al. (2018) AIRL：GAIL 拿不到 reward，AIRL 做了什么改动
- [ ] 动手：最小 GAIL 实现（需要 torch，届时另开 `code/deep/`）
- [ ] 动手：转移性实验——换掉 dynamics，看学到的 reward 还准不准

**验收**：能解释“GAIL 学不出 reward，AIRL 学得出”这句话到底在说什么。
**笔记**：`notes/04-...`

---

## 阶段 4：接回自己的研究方向

- [ ] reward 的可辨识性：Ng, Harada & Russell (1999) → Amin & Singh (2016) → Cao et al. (2021)
- [ ] Skalse et al. (2022) reward hacking：不可辨识性在什么时候会真的咬人
- [ ] Waugh et al. (2011) inverse equilibrium problem：IRL 的多方版本
- [ ] Hadfield-Menell et al. (2016) CIRL
- [ ] Christiano et al. (2017) / DPO (2023)：RLHF 和 IRL 的关系是什么
- [ ] **接回 `game-theory-for-LLM`**：QRE 的 $\lambda$ 估计和 MaxEnt IRL 是同一件事吗？
      如果是，能不能用同一份代码跑出一致的结果？

**验收**：写出一份“IRL 与 inverse game theory 的对照表”，两边的符号能互相翻译。
**笔记**：`notes/05-...`、`notes/06-...`

---

## 给自己的三条原则

1. **先写反例。** 一个方法“什么时候会坏掉”比“它怎么运作”更难也更值钱。
2. **能跑就跑。** 论文的式子抄进 `code/` 之前，先想清楚输入输出的 shape。
3. **不要跳过阶段 0。** 大部分的 IRL 卡关其实是 forward RL 没学扎实。

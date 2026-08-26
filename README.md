# irl-learning-notes — 逆向强化学习 (Inverse RL) 学习笔记

私人学习仓库，用来记录**我自己**学 IRL 的过程：从 demonstrations 反推 reward function。

```
正向 RL:   reward  ──优化──▶  policy / 行为
逆向 RL:   行为 (demonstrations) ──推断──▶  reward
```

姊妹仓库 [`game-theory-for-LLM`](https://github.com/xxy2026-student/game-theory-for-LLM)
做的是 inverse **game** theory（多方互动、从均衡反推 payoff）；
这里是 inverse **RL**（单一 agent、从轨迹反推 reward）。走到后面应该会接起来。

---

## 目前的状态

**已读 5 篇**，整理成一条线：**[`notes/README.md`](notes/README.md)**

| # | 论文 | 这一层做的事 |
|---|---|---|
| 1 | Ng & Russell, ICML 2000 | 写下“专家最优”的线性不等式 → 得到一个锥，不是一个解 |
| 2 | Abbeel & Ng, ICML 2004 | 不解 reward，改让 $\mu(\pi)$ 对上 |
| 3 | Ziebart et al., AAAI 2008 | 在满足匹配的分布里取最大熵 → 指数族 + MLE |
| 4 | Ziebart, Bagnell & Dey, ICML 2010 | 熵换成因果熵，修正随机环境下的偏差 |
| 5 | Finn, Levine & Abbeel, ICML 2016 | 用采样近似 $Z_\theta$，脱离 tabular |

**代码还是空的。** `code/` 只有一份“该实现什么”的说明——
自己写一遍 MaxEnt IRL 是阶段 2 的验收标准，还没做。

---

## 仓库结构

```
ROADMAP.md         学习路线：四个阶段，每阶段的验收标准
papers/README.md   阅读清单（依“该用什么顺序读”排序）
GLOSSARY.md        术语中英对照，读论文时查，遇到新的就往下加

notes/             论文笔记 —— 读完一篇写一篇
  _template.md     模板（先写“一句话”和“反例”，再写细节）

code/              自己的实现 —— 目前是空的，README 写了该实现什么
journal/           学习日志 —— 每次坐下来学了什么、卡在哪里
```

## 怎么用

1. 看 `ROADMAP.md` 决定接下来学什么。
2. 从 `papers/README.md` 挑一篇读，读完用 `notes/_template.md` 开一篇笔记。
   模板刻意把“**用一句话说清楚**”和“**它在什么情况下会失败**”放最前面：
   能写出这两格，才算真的读懂了。
3. 能写成代码的就写进 `code/`，并补一条测试。
   *跑得起来的 100 行 > 看起来很懂的 10 页笔记。*
4. 在 `journal/` 记下卡住的地方——过两周回来看，那些才是真正学到东西的位置。

## 进度

细项见 `ROADMAP.md`。**打勾的只有“读完并写出笔记”的部分，实现全部还没动。**

- 阶段 0：forward RL 前置 —— Bellman、soft Bellman 已从论文里读到；**value iteration 还没自己写过**
- 阶段 1：经典 IRL —— ✅ Ng & Russell 2000、Abbeel & Ng 2004
- 阶段 2：最大熵框架 —— ✅ Ziebart 2008、2010；**自己实现还没做**
- 阶段 3：对抗式 / 深度 IRL —— ✅ Finn 2016（GCL）；GAIL / AIRL 未读
- 阶段 4：接回 inverse game theory 与 RLHF —— 未开始

下一步（见 `notes/README.md` §8）：把 Abbeel & Ng 那两个 bound 的常数核对回论文，
然后开始写 `code/`——阶段 2 的验收是自己的 MaxEnt IRL 能在 GridWorld 上还原出 reward。

## 环境

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

目标是全程**离线可跑**：不需要 API key、不需要 GPU。
前三个阶段用 NumPy 就够了，到 GAIL 才会需要深度学习框架。

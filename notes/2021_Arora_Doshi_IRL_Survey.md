# A Survey of Inverse Reinforcement Learning: Challenges, Methods and Progress

- **作者 / 出处**：Saurabh Arora, Prashant Doshi · *Artificial Intelligence* **297**: 103500 (2021)
  · [arXiv:1806.06877](https://arxiv.org/abs/1806.06877)（2018 首发，多次修订后定稿）· [DOI](https://doi.org/10.1016/j.artint.2021.103500)
- **读的日期**：2026-08-31
- **状态**：📖 导读（PDF 待入库；本文档按主题走读，**章节编号以 PDF 为准**）

---

## -1. 通用 IRL 综述的真实版图（以及为什么带读这篇）

只算**通用**综述（不算 ICRL / LLM / 迁移这些子方向切面）,三代：

| 代际 | 综述 | 出处 | 定位 |
|---|---|---|---|
| 第一代 | Shao & Er, *A review of IRL theory and recent advances* | [IEEE CEC 2012](https://ieeexplore.ieee.org/document/6256507/) | 史料价值 |
| 第二代 | **Arora & Doshi**（本篇） | AIJ 2021 ([1806.06877](https://arxiv.org/abs/1806.06877)) | **公认主参考**：问题—挑战—方法分类学最完整 |
| 第二代 | Adams, Cody & Beling | [AIR 55:4307–4346 (2022)](https://doi.org/10.1007/s10462-021-10108-x) | 面向 deep-IRL 时代读者的重述，工程视角更多 |
| 第三代 | *Advances and applications in IRL: a comprehensive review* | [Neural Comput & Applic 37:11071–11123 (2025)](https://doi.org/10.1007/s00521-025-11100-0) | **最新的通用综述**，应用导向；Springer 付费墙 |
| 第三代·中文 | 《逆强化学习算法、理论与应用研究综述》 | [自动化学报](https://www.sciengine.com/AAS2/doi/10.16383/j.aas.c230081) | 中文系统梳理，可作平行参考 |

为什么主读 2021 这篇而不是 2025 那篇：NCA 2025 应用导向、影响力尚未建立，且获取受限；
Arora & Doshi 是所有后续文献引用的坐标系——**你书架上 `前沿文献/` 六个文件夹，
几乎每个都是这篇综述"挑战清单"里某一条在 2021 之后长成的子领域**（见 §6）。
读它不是读旧闻，是拿到整个领域的目录树；缺的年轮由本文 §6 补齐。

## 0. 一句话

> 给定 MDP∖R 和专家行为（轨迹或策略），求"最能解释该行为"的 reward——
> 这个问题在四十年里长出四族方法（间隔、熵、贝叶斯、分类回归），
> 但所有方法都在跟同五个结构性困难缠斗：解不唯一、观测不完美、要泛化、
> 依赖先验假设、内层 RL 太贵；综述的价值就是把方法学按"各自缓解哪个困难"摆上桌。

## 1. 它在解什么问题（综述本身的组织问题)

不是又一个方法清单。它回答三个元问题：

1. **IRL 的规范陈述是什么**——输入输出、解概念（§2）；
2. **为什么难**——五个结构性挑战（§3，全文最值钱）；
3. **每族方法用什么推断手段、付出什么假设**（§4-5）。

我们五篇笔记是沿时间轴的"纵读"；这篇综述提供"横读"：同一时刻把所有分支摆开。
纵读已完成的人做横读，收益集中在**没走过的分支**：贝叶斯族、博弈式间隔法、分类回归族、
遮挡/多意图扩展——下面重点讲这些。

## 2. 规范陈述与记号对齐

综述的设定：MDP∖R $=(S,A,T,\gamma)$ 已知（转移 $T$ 已知！），
专家输入是完整策略 $\pi_E$ 或示范集 $\mathcal D=\{\tau_i\}$，
求 $\hat R$ 使 $\pi_E$ 在 $\hat R$ 下（近似）最优。

| 综述用 | 我们 README 用 | 备注 |
|---|---|---|
| $T(s,a,s')$ | $P_a$ | 转移 |
| $\hat R$ | $R$ 或 $\theta^\top\phi$ | 常设线性特征 |
| ILE / EVD | —— | 评价指标，见 §5.3 |

留意它对 **IRL vs. apprenticeship** 的区分：目标是 reward 本身（可迁移、可解释）
还是只要一个不差于专家的策略。我们第 2 篇（Abbeel & Ng）就是从前者退到后者；
GAIL 走到纯 apprenticeship 的极端，AIRL 又拉回来——综述把这条摆动写成主线之一。

## 3. 五大挑战（读综述先读这章，其余按需查）

1. **不适定 / 解不唯一**：$R\equiv0$ 与整个锥都是解——即我们 §1 推出的 $(\star)$。
   综述的贡献是把各方法重新读作"往锥里加什么偏好"：间隔法加"分得开"，
   熵法加"少承诺"，贝叶斯加"先验"。
2. **不完美观测下的准确推断**：轨迹有噪声、不完整、被**遮挡**（occlusion，
   Doshi 组自己的战场：巡逻机器人只能看见对方轨迹的一段）；专家本身次优。
3. **泛化性**：$\hat R$ 要在示范没覆盖的状态、甚至新动力学下有效
   （AIRL 的 disentangled reward 是这条的后续）。
4. **对先验知识的敏感**：特征 $\phi$、已知 $T$、$\gamma$ ——错设都会整体传导。
   "reward 线性于手工特征"在 2016 前是全领域的隐藏假设。
5. **复杂度**：内层要反复解 MDP（每试一个候选 $R$ 解一次 forward 问题），
   状态空间一大就炸。后续所有"单层化"工作（IQ-Learn、DPO）都是在拆这条。

**带着读**：每见一个方法，先问"它压的是哪条，代价转嫁给了哪条"。
例：MaxEnt 压 1（分布上取最大熵），代价在 5（要算配分函数）。

## 4. 四族方法（按推断手段；命名以正文为准）

### 4.1 间隔优化（我们已读大半）

Ng & Russell 的 LP、Abbeel & Ng 的投影法之外，两个我们没读过的成员值得记：

- **MMP**（Ratliff, Bagnell & Zinkevich 2006）：结构化预测视角，
  最小化 $\tfrac12\lVert w\rVert^2+\sum_i\big[\max_{\mu\in\mathcal G_i}\big(w^\top\mu+\ell_i(\mu)\big)-w^\top\mu_i\big]$
  ——**损失增广**的 hinge：不仅要专家最优，还要以 margin 优于"按损失加权"的替代路径。
  次梯度 = 对损失增广 MDP 做一次规划。把 IRL 变成 structured SVM，工程上第一次能进机器人。
- **MWAL**（Syed & Schapire 2008）：把 apprenticeship 写成**零和博弈**：
  $$v^*=\max_{\psi\in\Delta(\Pi)}\ \min_{w\in\Delta_k}\ w^\top\big(\mu(\psi)-\hat\mu_E\big)$$
  min 玩家在特征单纯形上挑最坏 reward，max 玩家混合策略；乘性权重迭代求解。
  与投影法的关键差别：保证"**最坏 reward 下也不差于专家**"，且当博弈值 $v^*>0$ 时
  **严格超过**专家——第一个能原则性超越示范者的方法。
  （这行直通 game-theory-for-LLM：把对齐写成 policy 与 reward 的极小极大博弈，
  正是 Nash-LHF / adversarial training 的雏形。）

### 4.2 熵优化（我们的主线，略）

MaxEnt(08) → MaxCausalEnt(10) → REIRL（Boularias 2011：相对熵版本，
基线分布 + 重要性采样，**免模型**地做熵族 IRL，是 GCL 采样思想的前奏）→ GCL/GAIL/AIRL。
综述成稿早，对抗线（GAIL 之后）覆盖较浅——用你 `基础文献/` 的原文补。

### 4.3 贝叶斯族（书架上缺的一支，精读优先级最高）

**BIRL**（Ramachandran & Amir 2007）：把 $R$ 当随机变量，
$$P(R\mid\mathcal D)\ \propto\ P(\mathcal D\mid R)\,P(R),\qquad
P(\mathcal D\mid R)=\prod_{(s,a)\in\mathcal D}\frac{e^{\alpha\,Q^*_R(s,a)}}{\sum_{a'}e^{\alpha\,Q^*_R(s,a')}}$$

| 符号 | 意义 | 备注 |
|---|---|---|
| $\alpha$ | 理性温度 | $\alpha\to\infty$ 退化为"严格最优"约束 |
| $Q^*_R$ | $R$ 下的最优 Q | 每个候选 $R$ 都要解一次 forward——挑战 5 满额 |
| 先验 $P(R)$ | 高斯/拉普拉斯/…… | 显式接管"从锥里挑哪点"的准则 |

推断用 **PolicyWalk**（reward 空间上的 MCMC，相邻样本间增量修补 $Q$）；
**MAP-BIRL**（Choi & Kim 2011）改为对后验做梯度上升，统一了一批方法
（MaxEnt、MMP 都可写成"某似然+某先验"的 MAP——综述里这张统一表值得抄下来）；
**GPIRL**（Levine et al. 2011）给 $r(\phi)$ 挂 GP 先验，第一批非线性 reward。

**两个值得咀嚼的点**：
- 输出是**后验分布**而非点估计——"解不唯一"不再是 bug 而是 posterior 的宽度。
  Ng 2000 的锥 ≈ $\alpha\to\infty$、均匀先验下后验的支撑集。
  这条线 2021 后被频率派接管，变成 `01_Offline_IRL` 的可行集估计。
- **BIRL 的 Boltzmann 用硬 $Q^*$**（先算最优再加噪声），
  **MaxCausalEnt 的策略是 soft-Q 的 softmax**（噪声内生于规划）。
  两个似然不同——很多文献混用"Boltzmann rationality"一词，读时要分辨。

### 4.4 分类 / 回归族（免内层 RL 的旁路）

**SCIRL**（Klein et al. 2012）：把专家的 $(s,a)$ 当分类数据，
打分函数参数化为 $Q_w(s,a)=w^\top\mu(s,a)$（特征期望当基函数）——
分类器学到的决策函数**本身就是某个 reward 的贪心策略**，绕开 forward 求解；
**CSI** 级联版。假设强（要好的特征期望估计），但把挑战 5 直接归零——
这个"不解 RL 也能做 IRL"的火种后来在 Swamy 2023 和 IQ-Learn 里复燃。

## 5. 扩展专题与评价

### 5.1 不完美观测
遮挡下的 IRL（只见轨迹片段：对隐藏段取期望，EM 结构）；示范者与观察者视角不一致。
这支是 Doshi 组主场，综述覆盖最全面——机器人巡逻、多机器人设定。

### 5.2 多意图 / 多专家
一批示范来自多个 reward：EM 聚类 + IRL（Babeş-Vroman et al. 2011）、
非参贝叶斯 DPM-BIRL（Choi & Kim 2012）。→ 你的 `06_Latent与Switching_Reward_IRL`
正是这支的现代形态（意图还会中途切换）。

### 5.3 评价指标
**EVD**（expected value difference）：$V^{\pi^*_{R}}_{R}-V^{\pi^*_{\hat R}}_{R}$，
用真 reward 给学出的策略打分；**ILE**（inverted learning error）类似。
注意：EVD 度量的是 apprenticeship 成功，**不是 reward 恢复成功**——
两个 reward 可以 EVD 相同而迁移行为截然不同（AIRL 的动机）。
综述抱怨领域缺统一 benchmark——2026 年了这条依然成立，动手时自建对照仍是常态。

## 6. 综述什么时候"坏掉" ⭐（2020 → 2025 补丁）

内容冻结在 ~2020。它的挑战清单没有过时，但每条挑战之后都长出了它没来得及写的子领域——
恰好一一对上你的书架：

| 综述的挑战/伏笔 | 2021 后长成 | 你的书架 |
|---|---|---|
| 挑战 1 解不唯一 | 可行集/等价类的样本复杂度理论（Metelli 21/23、Lazzati 24/25、Zhao 24、Cao 21 可辨识性） | `01_Offline_IRL` + `基础文献/2021_Kim`、`2023_Metelli` |
| 挑战 2 准确推断 | 主动选询问/设计环境来收缩可行集 | `02_Active_IRL与Environment_Design` |
| 挑战 5 内层 RL 贵 | 单层化：IQ-Learn（Q 空间双射）、Swamy 2023 *IRL without RL*（免全局内层求解） | `基础文献/2021_Garg_IQ_Learn` |
| 多专家扩展 | 博弈化：多智能体 IRL、逆博弈 | `03_Multi-Agent_IRL与Inverse_Games` |
| 熵族的轨迹分布视角 | 直接学轨迹分布，绕开 reward：diffusion/flow 策略 | `04`、`05` 文件夹 |
| 多意图扩展 | 潜变量/切换 reward | `06_Latent与Switching_Reward_IRL` |
| （完全缺席） | 偏好数据大规模化：RLHF/DPO，即 LLM 对齐 | `notes/2025_Sun_IRL_meets_LLM_Survey.md` |
| （完全缺席） | 反推约束而非 reward：ICRL | [TMLR 2025 综述](https://arxiv.org/abs/2409.07569) |

另外两条局限：对抗式 IL（GAIL 之后）着墨少；应用清单已老（自动驾驶/机器人之外，
如今最大应用是对齐）。前者用原始论文补，后者若需要再看 NCA 2025。

## 7. 和已经读过的东西怎么接

- 综述 §3 挑战 1 ⇔ 我们 §1 的锥 $(\star)$；挑战 5 ⇔ 我们 §5 里 $Z_\theta$ 的采样近似动机。
- MWAL 的零和博弈 ⇔ GAIL 的 min-max 的"前对抗时代"版本；也是 game-theory-for-LLM 的入口。
- BIRL 后验宽度 ⇔ `01` 文件夹可行集的贝叶斯前身；先验 $P(R)$ ⇔ Ng 2000 的 $\lambda\lVert R\rVert_1$ 项。
- SCIRL"不解 RL" ⇔ IQ-Learn、DPO 的单层化谱系起点。

## 8. 动手

- [ ] **PolicyWalk 上 GridWorld**（书架上没有的分支里，唯一一个下午能写完的）：
  均匀先验 + Boltzmann 似然 + reward 空间随机游走；
  画后验样本的投影，肉眼确认"后验支撑 ≈ Ng 的锥"；
  再把 $\alpha$ 从 0.5 扫到 10，看后验从弥散到收缩进锥。
- [ ] 顺手比较：同一批示范，MaxEnt 点估计落在 BIRL 后验的什么位置（预期：高密度区，但不是众数——似然不同）。

## 9. 留着的问题

- BIRL 的 per-(s,a) 独立似然和 MaxCausalEnt 的轨迹似然，在什么极限下给出同一后验？
- 综述期待的"统一 benchmark"至今没有——可行集理论（`01`）能否给出"与算法无关"的评价：直接度量估计集与真实可行集的 Hausdorff 距离？
- MWAL 的"最坏 reward 下超过专家"与悲观离线 IRL 的最坏情形,是同一个 minimax 吗？形式化写一遍。

---

## 附：获取方式与阅读路径

- **PDF**：[arXiv:1806.06877](https://arxiv.org/abs/1806.06877)（免费终版）；正式版 [AIJ DOI](https://doi.org/10.1016/j.artint.2021.103500)。
  下载后放 `paper/综述/2021_Arora_Doshi_IRL_Survey.pdf`（`paper/综述/README.md` 已建好书架）。
  ⚠️ 本导读在无法访问 arXiv 的环境里按主题写成，式子与事实逐一核对过，
  但章节编号、方法族命名以 PDF 为准；如有出入以正文为准并回来修订。
- **两遍法**：第一遍读挑战章 + 每族方法的开头（半天，重点 §4.3 贝叶斯族）；
  第二遍只精读 BIRL/MAP-BIRL 与 MWAL 两小节 + 扩展章的遮挡与多意图（半天）。
  间隔法与熵族你已从原文读过，综述里只需看它的"统一 MAP 表"。

# Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities

- **作者 / 出处**：Hao Sun, Mihaela van der Schaar（Cambridge, van der Schaar Lab）· [arXiv:2507.13158](https://arxiv.org/abs/2507.13158)（2025-07-17）
- **配套材料**：同名 ACL 2025 tutorial（[ACL Anthology](https://aclanthology.org/2025.acl-tutorials.1/) · [slides 站点](https://sites.google.com/view/irl-llm)）
- **读的日期**：2026-08-31
- **状态**：📖 导读（PDF 待入库，见文末"获取方式"）

---

## -1. 为什么选这篇（综述生态位）

近几年 IRL 相关的综述，按定位排一下：

| 定位 | 综述 | 出处 | 适合什么时候读 |
|---|---|---|---|
| 通用·经典 | Arora & Doshi, *A survey of IRL: Challenges, methods and progress* | AIJ 2021 ([1806.06877](https://arxiv.org/abs/1806.06877)) | **通用综述的主参考**，精读导读见 [2021_Arora_Doshi_IRL_Survey.md](2021_Arora_Doshi_IRL_Survey.md) |
| 通用·深度方法 | Adams, Cody & Beling, *A survey of IRL* | [Artif. Intell. Review 2022](https://dl.acm.org/doi/10.1007/s10462-021-10108-x) | 想补 deep IRL 工程细节时 |
| 子方向·学约束 | Liu et al., *A Comprehensive Survey on Inverse Constrained RL* | TMLR 2025 ([2409.07569](https://arxiv.org/abs/2409.07569)) | 读完 `01_Offline_IRL` 后想看"反推约束而非 reward" |
| 邻域·正向 reward | *Reward Models in Deep RL: A Survey* | IJCAI 2025 ([2506.15421](https://arxiv.org/abs/2506.15421)) | 对照：forward RL 里 reward 怎么设计 |
| **本篇** | Sun & van der Schaar, *IRL Meets LLM Post-Training* | arXiv 2025-07 ([2507.13158](https://arxiv.org/abs/2507.13158)) | **IRL 视角下的 LLM 对齐**，目前最新、且正好接上 game-theory-for-LLM 那条线 |

选这篇的理由：它不是"又一篇 IRL 方法分类学"，而是回答一个新问题——
**过去五年 IRL 最大的应用场景（LLM post-training）里，经典 IRL 的哪些思想换了名字活着，哪些假设失效了。**
对已经读完 Ng 2000 → Finn 2016 这条线的人，这篇是把存量知识"变现"的最短路径。

副标题就是全文结构：**Basics**（§2-3 前后）→ **Advances**（中段）→ **Opportunities**（末段），
外加一节非常工程的 practical aspects（数据集/评测/infra）。下面按这四块走。

---

## 0. 一句话

> LLM 的 post-training 表面上是 RL，骨子里是 IRL：语言任务没有天然 reward，
> 所有 reward 都得从人类数据（示范、偏好、验证器）里**学**出来——
> 所以对齐流水线的每一段（SFT / RLHF / DPO）都能在经典 IRL 里找到前身，
> 而经典 IRL 的老病（解不唯一、分布偏移）也换上新名字（reward hacking、过优化）回来了。

## 1. 它在解什么问题

我们的五篇线停在 Finn 2016：用采样近似配分函数 $Z_\theta$，IRL 走出 tabular。
之后主线在机器人侧继续（GAIL → AIRL → IQ-Learn，`基础文献/` 里都有），
但 2022 年后 IRL 思想真正的主战场变成了 LLM——只是没人再叫它 IRL：

- 叫 **reward modeling** 的，是在做"从偏好反推 reward"；
- 叫 **RLHF** 的，是 preference-based IRL + 一步 KL 正则的 policy optimization；
- 叫 **DPO** 的，是把 reward 参数化进 policy 里——IQ-Learn 对 MaxEnt-IRL 干过一模一样的事。

这篇综述做的就是把这层窗户纸捅破，给出统一记号下的对应表，
并且认真讨论 **LLM 这个 MDP 和经典 IRL 的 MDP 哪里不一样**——
这决定了哪些经典结论能搬、哪些不能。

## 2. 读前先对齐：LLM 是个什么样的 MDP

综述强调的第一件事（也是全文最有信息量的一张对照）。记号沿用我们 README 的约定，新增：

| 符号 | 意义 | 对应经典 IRL 里的 |
|---|---|---|
| $x$ | prompt | 初始状态 $s_0$（从 $\mathcal D$ 抽） |
| $y=(y_1,\dots,y_T)$ | 一条回复 | 轨迹 $\tau$ |
| $s_t=(x,y_{<t})$ | 上下文 | 状态（**含全部历史**） |
| $a_t=y_t\in\mathcal V$ | 下一个 token | 动作，$\lvert\mathcal V\rvert\sim10^5$ |
| $\pi_{\rm ref}$ | SFT 后的参考模型 | ——（经典 IRL 没有这个东西！） |
| $r(x,y)$ | 整条回复的 reward | 轨迹级 reward（终端稀疏） |
| $\beta$ | KL 正则强度 | MaxEnt 里的温度 $1/\alpha$ |

和我们熟悉的设定比，五个关键差异：

1. **转移是已知且确定的**：$s_{t+1}=s_t\oplus a_t$（拼接）。
   Ng & Russell 需要知道 $P_a$，Ziebart 2010 花力气处理随机转移——这些困难在 LLM 里**直接消失**。
   MaxEnt(2008) 和 MaxCausalEnt(2010) 的区别在这里不存在，因为没有环境随机性。
2. **reward 天然是轨迹级、终端稀疏的**：人类比较的是整条回复。
   credit assignment（哪个 token 该负责）成为核心难题——这是 process reward model 存在的理由。
3. **动作空间巨大但有强先验**：$\pi_{\rm ref}$ 携带了预训练的全部知识。
   经典 IRL 从零学 policy；LLM 对齐是在一个已经很好的 policy 附近**微调**。
   KL 正则不是可选技巧，是防止灾难性偏离先验的结构件。
4. **纯离线偏好数据为主**，但"环境"（生成）可以无限模拟——
   和 `01_Offline_IRL` 的离线设定不同：那里贵的是环境交互，这里贵的是**人类标注**。
5. **示范者不是"最优专家"**：标注员在两条候选里挑较好的那条，
   两条可能都很差。Ng 2000 的缺口 (b)（专家次优 → LP 无解）在这里是常态而非例外，
   所以整个领域从一开始就建立在概率偏好模型（下面的 BT）上，而不是最优性不等式上。

**读综述时带着的问题**：每见到一个 LLM 技巧，先问"它利用了上面哪条退化性"。

## 3. Basics 主干：三个式子串起 SFT / RLHF / DPO

### 3.1 SFT = Behavior Cloning

$$\max_\theta\ \mathbb E_{(x,y)\sim\mathcal D_{\rm demo}}\big[\log\pi_\theta(y\mid x)\big]$$

就是 BC。经典结论照搬：分布偏移 + 复合误差（Ross & Bagnell 的 $O(\epsilon H^2)$），
体现为 LLM 的 exposure bias / 一步错步步错。
**IRL 存在的第一性理由**（学 reward 才能在示范分布外泛化）原封不动地适用。

### 3.2 RLHF = Bradley-Terry 反推 reward + KL 正则的 soft RL

**第一步（这一步才是 IRL）**：从成对偏好学 reward。Bradley-Terry 模型：

$$P(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big),\qquad
\mathcal L_{\rm RM}(\phi)=-\mathbb E\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$

| 符号 | 意义 | 备注 |
|---|---|---|
| $y_w,y_l$ | 同一 prompt 下被偏好/被拒的回复 | 标注单位是**比较**，不是打分 |
| $\sigma$ | logistic 函数 | BT = 差值上的逻辑回归 |
| $r_\phi$ | reward model，通常是 LLM 换头 | 训练完**冻结**，当真 reward 用 |

**不可辨识性又来了**（Ng 2000 的锥，换了形状）：
$r_\phi(x,y)\mapsto r_\phi(x,y)+c(x)$ 不改变任何 BT 概率——
逐 prompt 的平移方向完全学不到。

**第二步（forward RL）**：拿冻结的 $r_\phi$ 优化 policy，带 KL 约束：

$$\max_\pi\ \mathbb E_{x\sim\mathcal D,\ y\sim\pi(\cdot\mid x)}\big[r_\phi(x,y)\big]
-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\rm ref}(\cdot\mid x)\big)$$

这个目标有闭式解（变分法，逐 prompt 独立）：

$$\boxed{\ \pi^*(y\mid x)=\frac{1}{Z(x)}\,\pi_{\rm ref}(y\mid x)\,
\exp\!\big(r_\phi(x,y)/\beta\big)\ },\qquad
Z(x)=\textstyle\sum_y \pi_{\rm ref}(y\mid x)\,e^{r_\phi(x,y)/\beta}$$

**盯着这个式子看**：把 $\pi_{\rm ref}$ 换成均匀分布、$\beta=1$，
就是 Ziebart 2008 的 $P(\tau)\propto e^{\theta^\top f_\tau}$。
RLHF 的最优 policy 是 **MaxEnt IRL 的轨迹分布，先验从均匀换成 $\pi_{\rm ref}$**。
配分函数 $Z(x)$ 逐 prompt 一个、要靠从 $\pi_{\rm ref}$ 采样近似——
这正是 Finn 2016 用重要性采样近似 $Z_\theta$ 的手法。我们五篇的两个终点在这里会师。

（实操上第二步用 PPO；综述会讲后来的简化：GRPO 之类用组内均值当 baseline 省掉 value network。细节读到再说，骨架是上面两步。）

**注意那个"幸运的巧合"**：BT 学不到的方向 $c(x)$，代进闭式解正好被 $Z(x)$ 吸收，
$\pi^*$ 不变。**偏好数据不可辨识的成分 = policy 优化不需要的成分。**
这是 LLM 设定对 IRL 病态性的第一重豁免（但只对轨迹级 reward 成立——对 process reward 不成立，见 §4.2）。

### 3.3 DPO = 把 reward 代换掉（IQ-Learn 的翻版）

对闭式解取对数、反解 $r$：

$$r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\log Z(x)$$

代回 BT 损失，$Z(x)$ 在差值里**相消**：

$$\mathcal L_{\rm DPO}(\theta)=-\mathbb E\Big[\log\sigma\Big(
\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\pi_{\rm ref}(y_w\mid x)}
-\beta\log\tfrac{\pi_\theta(y_l\mid x)}{\pi_{\rm ref}(y_l\mid x)}\Big)\Big]$$

两步并一步：不再显式学 $r_\phi$，policy 本身隐式参数化了一个 reward
$\hat r_\theta=\beta\log\frac{\pi_\theta}{\pi_{\rm ref}}$（"your LM is secretly a reward model"）。

**结构上和 IQ-Learn 完全同一个招**（`基础文献/2021_Garg_IQ_Learn.pdf`）：
IQ-Learn 利用 $r\leftrightarrow Q$ 的双射把 MaxEnt-IRL 的 min-max 变成单层优化；
DPO 利用 $r\leftrightarrow\pi^*$ 的双射（在 KL 正则下）把 RLHF 两步变成一步。
对比着读会非常快。

**代价也对称**：换掉显式 reward，就失去了"reward 可以拿去评估新样本 / 做 best-of-N / 迁移"
的能力（AIRL 强调过的 transferable reward，`2018_Fu_AIRL.pdf` 和 `2021_Metelli_Transferable_Rewards.pdf` 的主题）。
纯离线的 DPO 也重新暴露在分布偏移下——它从没见过 $\pi_\theta$ 自己生成的样本。

## 4. Advances：reward 信号的光谱 + 老病复发

### 4.1 reward 从哪来（综述中段的主体，按信号源分类）

| 信号源 | 代表 | IRL 视角 |
|---|---|---|
| 人类示范 | SFT | BC |
| 人类成对偏好 | BT-RM / DPO | preference-based IRL |
| AI 反馈 | RLAIF、Constitutional AI | 用 LLM 自己当标注员，蒸馏一个 reward |
| 可验证信号 | RLVR（数学/代码，R1 一系） | reward 是**程序**不是学出来的——退出 IRL，回到真 reward 的 RL |
| 过程监督 | PRM（逐步打分）vs ORM（只看结果） | credit assignment：把终端 reward 摊到 token/步骤上 |
| 生成式评审 | LLM-as-judge / GenRM | reward model 输出的是文字评价再折成分数 |

注意 RLVR 那行：**当 reward 可验证时，问题就不再是 IRL**。
综述的边界意识在这里：IRL 视角覆盖的是"reward 必须从行为数据构造"的那部分对齐问题——
恰恰是最难评测、最容易 hack 的部分（helpfulness、无害、风格）。

### 4.2 Reward hacking = 不可辨识性 + 分布偏移的现代形态

RM 只在标注数据的分布上被约束（有限比较 ⇒ reward 等价类很大，Ng 2000 的老话），
policy 优化却会**主动搜索** RM 的高分区域——正是数据分布外约束最弱的地方。
Goodhart：对 proxy 优化过猛，真实质量先升后降（RM 过优化的 scaling law，Gao et al. 2023 一线）。

这和 `01_Offline_IRL` 文件夹是同一个问题的两种口音：
那边的答案是悲观原则（对可行集里最坏的 reward 稳健），
这边的工程答案是 KL 约束 + 早停 + RM ensemble + 定期换新数据重训 RM。
**悲观 IRL 的理论和 RLHF 的工程实践至今没有真正接上——这是能写 paper 的缝。**

另外注意 §3.2 那个豁免的失效边界：process reward 在中间步骤上打分，
逐状态的平移 $c(s_t)$ 不再被 $Z$ 吸收，shaping 等价类的问题（AIRL §5 的主题）整个回来了。

### 4.3 推理时的 reward：不改权重也能"对齐"

RM 的第二用途：不训练，只在推理时用——best-of-N 采样、reranking、PRM 引导的树搜索。
IRL 视角：这是拿学到的 $r$ 做 **planning**（转移已知且确定，才可能纯靠搜索），
又一次吃了 §2 第 1 条退化性的红利。经典 IRL 里没有这个选项，因为转移未知。

## 5. Opportunities + 实践清单

末段两块：

- **practical aspects**：偏好数据集（HH-RLHF、UltraFeedback 一系）、RM 评测（RewardBench 一系）、
  训练 infra（TRL / OpenRLHF / veRL）与省算力技巧。这部分当**工具书**用，不精读，
  需要跑实验时回来查表。
- **从 sparse-reward RL 借镜的开放问题**：终端稀疏 reward 下的 credit assignment、
  exploration、reward shaping 怎么迁移到 LLM 推理训练。
  这块和 `06_Latent与Switching_Reward_IRL`（reward 不止一个、还会切换）能对上话。

## 6. 这篇综述什么时候会"坏掉" ⭐

对综述要问的不是反例，是**立场偏差和保质期**：

1. **"一切皆 IRL"是一种视角选择**。它照亮了 reward 学习这条线，但也有代价：
   BT 假设本身（偏好可以用一个标量 reward 的差解释）就值得怀疑——
   真实人类偏好有**非传递性、标注者异质性**。抛弃 BT 的路线
   （Nash learning from human feedback：直接在偏好博弈里找纳什均衡）
   天然是 game-theory-for-LLM 那个仓库的题目，综述以 BT 流水线为主线，这条线要自己延伸。
2. **保质期**：2025-07 的快照。RLVR / 推理模型一侧月更，读的时候把"方法名录"当地图别当领土；
   但 §3 的三个式子是结构性的，不会过时。
3. **理论搬运警告**：`01_Offline_IRL` 里的样本复杂度结果大多假设未知转移、可交互环境，
   在"转移已知确定 + 纯离线偏好"的 LLM 设定下**不能直接引用**，只能借直觉。

## 7. 和已经读过的东西怎么接

- `notes/README.md` §3（Ziebart 2008）→ 本篇 §3.2：同一个指数族，先验换成 $\pi_{\rm ref}$。
- `notes/README.md` §5（Finn 2016 采样近似 $Z$）→ RLHF 逐 prompt 估 $Z(x)$。
- `基础文献/2021_Garg_IQ_Learn.pdf` → DPO：同构的"代换掉 reward"技巧（§3.3）。
- `基础文献/2018_Fu_AIRL.pdf` + `2021_Metelli_Transferable_Rewards.pdf` → 显式 RM 对 DPO 的优势
  （可迁移、可复用），以及 shaping 等价类如何在 PRM 里复发（§4.2）。
- `前沿文献/01_Offline_IRL` → reward 等价类 / 悲观原则 vs reward hacking（§4.2）。
- `前沿文献/03_Multi-Agent_IRL与Inverse_Games` → 非 BT 偏好、偏好博弈的纳什解（§6.1）。

## 8. 动手

- [ ] **tabular 版 DPO 验证**（GridWorld 精神续作，CPU 可跑）：
  "语言" = 长度 $\le 8$ 的 01 串，$\pi_{\rm ref}$ = 均匀自回归；
  真 reward $r^*(y)$ = "11" 子串个数；用 BT 从 $r^*$ 采偏好对；
  (a) 训个小 RM，检查它与 $r^*$ 差多少个 $c(x)$；
  (b) 跑 DPO，检查 $\beta\log\frac{\pi_\theta}{\pi_{\rm ref}}$ 是否恢复 $r^*$（差平移）；
  (c) 把 $\beta$ 调小、训久，亲眼看 reward hacking（KL 爆炸、输出全是"111…"）。
- [ ] 玩具 best-of-N vs DPO：同一 RM，比较推理时选择和权重更新的样本效率。

## 9. 留着的问题

- 悲观 IRL（可行集最坏情形）能否给 RLHF 一个不靠 KL-系数玄学的过优化上界？
- PRM 的逐步分数和 max-causal-entropy 的逐步 reward 分解是不是同一个对象？
- 偏好非传递时，BT-RM 学到的是什么？（→ 社会选择 / Nash-LHF，接 game-theory-for-llm）
- DPO 的隐式 reward 在 $\pi_\theta$ 低概率区域行为如何？（显式 RM 的 OOD 问题换了宿主）

---

## 附：获取方式与阅读路径

- **PDF**：[arXiv abs](https://arxiv.org/abs/2507.13158) / [PDF 直链](https://arxiv.org/pdf/2507.13158) / [HTML 版](https://arxiv.org/html/2507.13158)。
  下载后放 `paper/前沿文献/07_IRL与LLM对齐/2025_Sun_IRL_LLM_Survey.pdf`（README 已建好占位表）。
  ⚠️ 本导读在无法访问 arXiv 的环境里写成：结构与式子按主题梳理并逐一核对过，
  但**章节编号以 PDF 为准**；若正文与导读有出入，以正文为准并回来修订本文件。
- **建议路径**（两遍法）：
  第一遍只读 Basics + 每节开头段，对着本文 §2-3 核对式子（半天）；
  第二遍挑 Advances 里 reward hacking 与 PRM 两节精读，practical 清单跳过（半天）；
  slides（[tutorial 站点](https://sites.google.com/view/irl-llm)）适合第一遍前热身。

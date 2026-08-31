# 逆强化学习综述(2000–2026):从一个奖励,到一族奖励

**Inverse Reinforcement Learning: A Survey from Point Estimates to Reward Sets, 2000–2026**

- **写作时间**:2026-08-31
- **写作方式与可信度**:通用 IRL 综述自 Arora & Doshi (AIJ 2021) 与 Adams et al. (AIR 2022) 后长期缺位,
  本文为填补 2020–2026 缺口而自撰。素材三来源,文中引用按最弱一环标注:
  - **★** 本仓库 `paper/` 内有 PDF,且本次写作直接读过原文(至少摘要与导言);
  - **☆** 写作者训练语料内的文献(截至 2026-01),内容按记忆重述、要点经检索核对;
  - **◐** 仅读过摘要或检索确认存在(多为 2026 年新文),结论转述自摘要,**引用前请自行核原文**。
- **怎么读**:§1–2 是骨架(记号、三主轴、解概念),必读;§3–4 已读过原文的人可当校对;
  §5–10 是 2021 年后各分支,按兴趣选读;§13 开放问题面向找题。

---

## 目录

1. [问题、记号与三条主轴](#1-问题记号与三条主轴)
2. [不适定性:从缺陷到研究对象](#2-不适定性从缺陷到研究对象)
3. [经典四族(2000–2012)](#3-经典四族2000-2012)
4. [深度与对抗时代(2015–2021)](#4-深度与对抗时代2015-2021)
5. [理论时代:可行集的样本复杂度(2021–2026)](#5-理论时代可行集的样本复杂度2021-2026)
6. [数据获取轴:主动 IRL 与环境设计](#6-数据获取轴主动-irl-与环境设计)
7. [多智能体与逆博弈](#7-多智能体与逆博弈)
8. [奖励之外的结构:约束、多意图、效用](#8-奖励之外的结构约束多意图效用)
9. [偏好时代:RLHF 即大规模 IRL](#9-偏好时代rlhf-即大规模-irl)
10. [生成策略时代:绕开奖励的模仿](#10-生成策略时代绕开奖励的模仿)
11. [评价与基准](#11-评价与基准)
12. [应用速览](#12-应用速览)
13. [开放问题](#13-开放问题)
14. [参考文献](#14-参考文献)

---

## 1. 问题、记号与三条主轴

### 1.1 形式陈述

给定 MDP∖R $=(S,A,P,\gamma)$(或有限视界版本)与专家行为的观测——
完整策略 $\pi_E$、示范轨迹集 $\mathcal D=\{\tau_i\}_{i=1}^m$、或更弱的信号(偏好、片段、遮挡轨迹)——
求能**解释**该行为的奖励 $R$:最强意义下 $\pi_E\in\arg\max_\pi J(\pi;R)$,
弱化意义下 $\pi_E$ 在 $R$ 下近似最优或 Boltzmann 理性。

记号沿用本仓库 `notes/README.md` 的约定并扩充:

| 符号 | 意义 |
|---|---|
| $P_a\in\mathbb R^{\lvert S\rvert\times\lvert S\rvert}$ | 动作 $a$ 的转移矩阵 |
| $\phi:S\to\mathbb R^k$,$f(\tau)=\sum_t\gamma^t\phi(s_t)$ | 特征与轨迹特征计数 |
| $\mu(\pi)=\mathbb E_\pi[f(\tau)]$,$\hat\mu_E$ | 特征期望及其经验估计 |
| $\rho_\pi(s,a)$ | 占用测度(discounted state-action visitation) |
| $\theta,w$ | 奖励参数(线性时 $R=\theta^\top\phi$) |
| $\mathcal R_{\rm feas}$ | 可行奖励集(§2.4) |
| $\Phi:S\to\mathbb R$ | shaping 势函数 |
| $\alpha,\ 1/\beta,\ 1/\lambda$ | 理性/熵温度(各文献记号不一,本文混用并随处注明) |
| $x,y,\pi_{\rm ref}$ | 仅 §9:prompt、回复、参考模型 |

### 1.2 三条主轴(本文的组织论点)

二十六年的文献可以摆在三条轴上;每个方法是三轴上的一个点,每个"时代"是一次沿某条轴的集体移动:

| 轴 | 起点(2000) | 终点(2026) |
|---|---|---|
| **A. 解概念** | 单点奖励 $\hat R$ | 奖励**分布**(贝叶斯)→ 奖励**等价类**(shaping/可辨识性)→ 奖励**可行集** $\mathcal R_{\rm feas}$ 及其集值估计 |
| **B. 数据假设** | 已知完整最优策略 $\pi_E$ | 有限轨迹 → 次优/Boltzmann 专家 → 成对偏好 → 遮挡/多意图/多环境/异质多示范者 → 主动选择数据 |
| **C. 计算结构** | 双层(外层猜 $R$,内层解 MDP) | 对抗式 min-max(GAIL/AIRL)→ **单层化**(IQ-Learn、DPO、免 RL 归约)→ 干脆绕开 $R$ 直接学轨迹/动作分布(§10) |

一句话版本:**IRL 的历史是"承认解不唯一"的历史(A),"承认数据不完美"的历史(B),和"拆掉内层 RL"的历史(C)。**

---

## 2. 不适定性:从缺陷到研究对象

### 2.1 锥与退化解(2000)

Ng & Russell ☆ 写下专家最优的充要不等式(设 $\pi_E(s)\equiv a_1$):

$$(P_{a_1}-P_a)\,(I-\gamma P_{a_1})^{-1}R\ \succeq\ 0,\qquad \forall a\neq a_1 \tag{$\star$}$$

解集是含 $R\equiv 0$ 的多面锥:线性齐次约束给出**可行域**而非解。
此后每一族方法都可读作"往锥里加一个选择准则"(§3),
而 2021 年后的理论(§5)则反过来把锥本身当作估计对象。

### 2.2 等价类:哪些方向注定学不到(1999–2018)

比"锥太大"更精细的问题是:**即使数据无限,哪些奖励差异原则上不可区分?**

- **Potential-based shaping**(Ng, Harada & Russell, ICML 1999 ☆):
  $$R'(s,a,s')=R(s,a,s')+\gamma\Phi(s')-\Phi(s)$$
  对任意 $\Phi$ 保持全体最优策略不变。shaping 方向是行为观测的"零空间"。
- **AIRL**(Fu et al., ICLR 2018 ★):对抗式 IRL 中显式把判别器参数化为
  $f_{\theta,\varphi}(s,a,s')=g_\theta(s)+\gamma h_\varphi(s')-h_\varphi(s)$,
  企图把"真奖励" $g$ 与 shaping 项 $h$ 解耦,换取跨动力学**可迁移**的奖励——
  等价类问题第一次被当作工程目标处理。
- **可辨识性刻画**(2021–2022):Kim et al. (ICML 2021) ★ 给出从最优行为恢复奖励的充要条件;
  Cao, Cohen & Szepesvári (NeurIPS 2021) ☆ 证明熵正则(Boltzmann)专家 + **已知温度**下,
  奖励可辨识到"常数 + shaping"——熵正则是统计上的祝福,这一发现贯穿后文
  (§7 的逆博弈、§9 的 RLHF 都靠它);Rolland et al. (NeurIPS 2022) ☆:
  **多环境**(同一奖励、不同动力学)的示范进一步切掉 shaping 方向;
  Skalse et al. (ICML 2023) ☆ 系统整理各数据类型诱导的"部分可辨识"格局。

### 2.3 一个反复出现的模式

> 单一环境、单一(近)最优专家 ⇒ 大等价类;
> 每引入一种"多样性"——熵噪声、多环境、多次优程度、多均衡、多意图——等价类就被切小一块。

§5–§8 的大部分定理是这个模式在不同设定下的实例。

### 2.4 可行集作为一等公民(2021–)

Metelli, Ramponi & Restelli (ICML 2021) ☆ 把问题重述为:估计**全部**可行奖励构成的集合
$\mathcal R_{\rm feas}$。在专家动作已知、动力学已知的确定专家情形,可行集有显式参数化:

$$R(s,a)\;=\;\underbrace{V(s)-\gamma\,\mathbb E_{s'\sim P(\cdot|s,a)}V(s')}_{\text{任意 }V\text{ 的 shaping}}\;-\;\underbrace{A(s,a)\,\mathbb 1\{a\notin \operatorname{supp}\pi_E(s)\}}_{A\ge 0:\ \text{非专家动作的罚}}$$

即:**可行集 = shaping 自由度 × 非专家动作上的非负优势罚**。
估计对象从点变成集合后,误差度量(Hausdorff 型)、样本复杂度、
悲观/乐观原则都随之改写——这是 §5 的主题。
2026 年的延伸:对**多个、次优程度各异**的示范者,把各自声明的次优水平写成线性约束、
对可行集取交,并刻画"新示范者何时严格收缩交集"(Kim, Deshmukh, Vlassis & Zhang, arXiv:2605.30903 ◐);
另有工作研究可行集的**闭式中心**(reward centroid)作为集值解的点代表(arXiv:2509.12010 ◐)。

---

## 3. 经典四族(2000–2012)

按推断手段分四族。逐族给出:核心式子、所在三轴位置、留下的缺口。
(前两族本仓库五篇笔记已精读,此处从简;后两族详一些。)

### 3.1 间隔优化

- **LP-IRL**(Ng & Russell, ICML 2000 ☆):$(\star)$ + 最大化专家与次优动作的 $Q$ 间隔 + $\ell_1$ 正则。
- **学徒学习/投影法**(Abbeel & Ng, ICML 2004 ☆):放弃恢复 $R$,匹配特征期望:
  $\lVert\mu(\tilde\pi)-\hat\mu_E\rVert_2\le\epsilon$ ⇒ 全体线性奖励下不差于专家 $\epsilon$。
  样本复杂度依赖 $k$ 而非 $\lvert S\rvert$。
- **MMP**(Ratliff, Bagnell & Zinkevich, ICML 2006 ☆):结构化 hinge、损失增广规划,IRL 进机器人。
- **MWAL**(Syed & Schapire, NeurIPS 2007 ☆):零和博弈
  $\max_{\psi\in\Delta(\Pi)}\min_{w\in\Delta_k}w^\top(\mu(\psi)-\hat\mu_E)$,
  乘性权重求解;最坏奖励下有保证,博弈值为正时**严格超过专家**。
  对抗式 IL(§4)与多智能体逆博弈(§7)的种子。

三轴位置:A=单点(或干脆不要 $R$);B=最优专家、被动轨迹;C=双层(内层反复规划)。

### 3.2 熵优化

- **MaxEnt**(Ziebart et al., AAAI 2008 ☆):在满足特征匹配的轨迹分布中取最大熵,
  $P(\tau)\propto e^{\theta^\top f_\tau}$;把"专家次优"吸收成似然噪声,把锥变成单峰凸问题。
- **MaxCausalEnt**(Ziebart, Bagnell & Dey, ICML 2010 ☆):随机转移下熵换因果熵,
  策略为 soft-Q 的 softmax:$\pi(a\mid s)=\exp(Q_{\rm soft}(s,a)-V_{\rm soft}(s))$,
  $V_{\rm soft}(s)=\log\sum_a e^{Q_{\rm soft}(s,a)}$。后来所有 soft-RL / RLHF 闭式解的原型。
- **REIRL**(Boularias, Kober & Peters, AISTATS 2011 ☆):相对熵 + 基线分布 + 重要性采样,
  熵族第一次**免模型**;Finn 2016 采样配分函数的前奏。

三轴位置:A=单点但由分布原则唯一化;B=有限轨迹、隐式容忍次优;C=双层(配分函数)。

### 3.3 贝叶斯族

- **BIRL**(Ramachandran & Amir, IJCAI 2007 ☆):$P(R\mid\mathcal D)\propto P(\mathcal D\mid R)P(R)$,
  似然为逐 $(s,a)$ 的 Boltzmann-of-$Q^*$:
  $P(\mathcal D\mid R)=\prod_{(s,a)}\frac{e^{\alpha Q^*_R(s,a)}}{\sum_{a'}e^{\alpha Q^*_R(s,a')}}$;
  PolicyWalk MCMC 采样。**输出是后验**——解不唯一变成后验宽度,轴 A 的第一次移动。
  注意与 3.2 的似然不同:BIRL 用硬 $Q^*$ 再加噪,MaxCausalEnt 噪声内生于规划。
- **MAP-BIRL**(Choi & Kim, NeurIPS 2011 ☆):对后验做梯度;MaxEnt、MMP 皆可写成
  "某似然 + 某先验"的 MAP——四族的第一张统一表。
- **GPIRL**(Levine, Popović & Koltun, NeurIPS 2011 ☆):GP 先验,非线性奖励。

缺口:每个候选 $R$ 内层解一次 MDP,轴 C 满额;后验的频率派对应物(置信集)等到 2021。

### 3.4 分类/回归族

- **SCIRL / CSI**(Klein et al., NeurIPS 2012 / ECML 2013 ☆):把专家 $(s,a)$ 当分类数据,
  打分函数参数化为 $Q_w=w^\top\mu(s,a)$,分类器的决策函数即某奖励的贪心策略——
  **完全绕开内层 RL**。假设强、影响当时不大,但它是轴 C"单层化"的最早火种
  (IQ-Learn、DPO 的精神祖先)。

---

## 4. 深度与对抗时代(2015–2021)

### 4.1 采样配分函数与深度奖励

**GCL**(Finn, Levine & Abbeel, ICML 2016 ☆):cost 网络 + 用当前策略的样本做重要性采样估计
$Z_\theta$,与策略优化交替;IRL 首次端到端进像素/连续控制。
同期 Wulfmeier et al. 的深度 MaxEnt(tabular 规划内层)☆。
Finn et al. 2016b ☆ 指出 GCL ≍ GAN ≍ EBM 训练的三位一体,直接引出:

### 4.2 对抗式模仿:占用测度视角

**GAIL**(Ho & Ermon, NeurIPS 2016 ☆)把"IRL 再 RL"复合过程刻画为占用测度匹配:

$$\text{RL}\circ\text{IRL}_\psi(\pi_E)\;=\;\arg\min_\pi\;-H(\pi)+\psi^*(\rho_\pi-\rho_{\pi_E})$$

正则 $\psi$ 取 GAN 型即得判别器目标
$\min_\pi\max_D\ \mathbb E_{\pi}[\log D]+\mathbb E_{\pi_E}[\log(1-D)]-\lambda H(\pi)$。
奖励退场,轴 A 走到"不要 $R$"的极端;**f-散度统一**(Ghasemipour et al., CoRL 2019 ☆)
把 BC/GAIL/AIRL 写成选不同 $f$ 的散度最小化;
**moment matching 博弈框架**(Swamy et al., ICML 2021 ☆)进一步统一并给出模仿间隙的上下界,
把 2004 年的特征匹配与 2016 年的对抗训练收进同一个极小极大。

**AIRL**(Fu et al., ICLR 2018 ★)把奖励请回来:解耦参数化换可迁移性(§2.2)。
这条"要不要显式 $R$"的摆动持续到今天:§9 的 RLHF-vs-DPO、§10 的生成策略都是它的回声。

### 4.3 单层化

- **IAVI**(Kalweit et al., 2020 ☆):tabular Boltzmann 专家下奖励的**闭式**反解(逆动作价值迭代),
  后被多意图 IRL(§8.2)用作内环。
- **IQ-Learn**(Garg et al., NeurIPS 2021 ★):逆软 Bellman 算子
  $r(s,a)=Q(s,a)-\gamma\,\mathbb E_{s'}[V^\pi(s')]$ 是 $r\leftrightarrow Q$ 的双射,
  把 min-max 整个转进 Q 空间变**单层凹优化**;隐式奖励的代价是失去可迁移性(与 DPO 同构,§9)。
- **免 RL 的 IRL**(Swamy et al., ICML 2023 ☆):利用专家状态分布做 resets,
  把 IRL 归约到**局部**策略搜索,内层不再解全局 RL——轴 C 在理论侧的收尾。
- 2025–26 出现**信赖域/近端**风格的非对抗 IRL(Trust Region IRL, ICML 2026 ◐;
  proximal inverse reward optimization, arXiv:2509.23135 ◐),动机是拿回对抗训练放弃的单调改进保证。

---

## 5. 理论时代:可行集的样本复杂度(2021–2026)

把 §2.4 的集值问题放进统计学习框架:估计 $\hat{\mathcal R}$,
在某个集合间度量下逼近 $\mathcal R_{\rm feas}$。按信息获取方式分四种设定:

### 5.1 生成模型(simulator)设定

Metelli et al. (ICML 2021) ☆ 首次给出可行集估计的样本复杂度(并同时处理向新环境迁移的
TRAVEL 目标);Metelli, Lazzati & Restelli (ICML 2023) ★ 系统化 Hausdorff 型度量族、
给出 PAC 上界与 minimax 下界。结论基调:已知动力学结构下,
**估计整个可行集并不比估计单个最优策略贵出维度级**。

### 5.2 在线交互(无生成模型)

**AceIRL**(Lindner, Krause & Ramponi, NeurIPS 2022 ★):顺序交互、无 simulator 下
主动探索以收缩可行集(与 §6 交叠);Zhao, Wang & Bai (ICML 2024) ★ 的 **RLE**:
归约到 reward-free exploration,在更强度量下达到近最优样本复杂度。

### 5.3 纯离线

- **Lazzati, Mutti & Metelli (ICML 2024)** ★:离线可行集的新解概念与 IRLO/PIRLO 算法,
  悲观原则保证估计集**包含**真实可行集(不漏),代价是可能多包;
- **RLP**(Zhao, Wang & Bai, ICML 2024 ★):以"reward mapping"(可行集的生成函数)为估计对象,
  悲观 + 单策略集中系数,**首个**标准离线设定下多项式样本的 IRL;配套下界证明近最优。
  论文标题之问的答案:在他们的度量下,**IRL 不比标准 RL 难多少**。
- **CLARE**(Yue et al., ICLR 2023 ☆):保守模型基离线 IRL 的工程侧代表。

### 5.4 大状态空间

Lazzati, Mutti & Metelli (NeurIPS 2024) ★ 用**奖励相容性**(reward compatibility)框架
替代硬可行性,允许"有多不相容"的连续度量,CATY-IRL 在线性 MDP 中高效;
扩展版见 arXiv:2501.07996 ◐。函数逼近下的可行集理论仍属起步(§13)。

### 5.5 次优与异质专家

Poiani et al. (arXiv 2024) ☆:$\epsilon$-次优专家把等式约束松成不等式,可行集变胖;
Kim et al. (arXiv:2605.30903, 2026 ◐):多个异质次优示范者、各自次优度已声明 ⇒
线性约束交集,单调收缩 + "何时严格收紧"的精确刻画。
与偏好数据的联系(§9):T-REX(Brown et al., ICML 2019 ☆)早已示范
"次优轨迹 + 排序"可**外推**出超越示范者的奖励。

### 5.6 小结

> 轴 A 的终点:解概念 = 可行集;轴 B 的现状:最优专家假设已全面松绑;
> 但这些结果几乎全部生活在 tabular / 线性、**已知或可交互动力学**的世界——
> 与 §9–10 的大模型工程之间隔着一整条未修的桥(§13)。

---

## 6. 数据获取轴:主动 IRL 与环境设计

被动示范覆盖不足 ⇒ 可行集收不紧。让学习者**选择要看什么**:

- **查询准则的谱系**:早期启发式(Lopes et al. 2009;Brown et al. 2018;Kweon et al. 2023 ☆)
  → 信息论化。**PAC-EIG**(Bajgar et al., RLC 2025 ★)对学徒策略的**后悔**取期望信息增益,
  给出 Boltzmann(噪声)专家下主动 IRL 的第一个 PAC 保证;学奖励本身为目标时用 Reward-EIG。
- **主动探索**:AceIRL(§5.2)★ 在环境里探索而非向专家提问,二者正交可组合。
- **环境设计**(Kleine Büning, Villin & Dimitrakakis, ICML 2024 ★):更进一步,
  由学习者**设计专家所处的环境/任务**(ED-BIRL/ED-AIRL)——
  实验设计思想进 IRL;呼应 §2.3:多环境切等价类(Rolland et al.)在这里从"恰好有"变成"主动造"。

---

## 7. 多智能体与逆博弈

观测对象从"单智能体最优"变成"均衡",三件事同时变糟:
均衡非唯一、每人的最优依赖他人、奖励歧义可能**改变博弈性质**(合作↔竞争)。

- **可行集的多智能体化**(Freihaut & Ramponi, NeurIPS 2025 ★):Markov 博弈中刻画
  "使观测策略组为 Nash"的全部奖励;核心负结果:**单个观测均衡不足以约束可行集**,
  不同均衡诱导不同可行集,可造成 $(1-\gamma)^{-1}$ 量级的 Nash gap。
  引入熵正则 Markov 博弈 ⇒ 均衡唯一化,给出生成模型下的样本复杂度;
  可辨识性:一般和博弈仅平均意义可辨识,奖励线性可分
  ($R(s,a,b)=R_A(s,a)+R_B(s,b)$)时可辨识到加性常数。
- **熵正则 QRE 的置信集**(Liao et al., ICML 2025 ★):两人零和矩阵/Markov 博弈,
  quantal response 均衡 + 线性参数化下的可辨识条件;算法输出**全部**可行参数的置信集
  (单智能体 Cao/Rolland 熵正则祝福的博弈版)。
- **有限理性的祝福**(Wu, Shen, Fang & Xu, NeurIPS 2022 ★):逆 Stackelberg——
  跟随者 quantal response 反而使领导者效用可高效学习;完美理性才是统计噩梦。
- **极小极大形式**(Goktas et al., ICLR 2024 ★):逆多智能体学习写成生成对抗式
  min-max,多项式时间可解。
- 与 §9 的接口:偏好聚合的博弈化(Nash learning from human feedback,Munos et al., ICML 2024 ☆)
  把"标注者群体"当对局者——多智能体逆博弈与对齐正在合流。

> 主题重现:熵正则在单智能体切等价类(§2.2),在博弈里还额外**选均衡**——一石二鸟。

---

## 8. 奖励之外的结构:约束、多意图、效用

标准 IRL 把行为的全部结构塞进一个平稳标量 $R$。三条放松路线:

### 8.1 反推约束(ICRL)

专家 = 最优 + **服从隐式约束**(Malik et al., ICML 2021 ☆)。
学出的约束(而非奖励)可迁移到新任务保安全。
系统综述见 Liu et al. (TMLR 2025) ☆:确定/随机环境、有限示范、多智能体下的约束推断框架。
与 §5 同构:约束的"可行集"同样病态,同样在做集值估计。

### 8.2 多意图与切换奖励

谱系(本仓库 `06` 文件夹主线):

| 年代 | 工作 | 意图结构 | 备注 |
|---|---|---|---|
| 2011 | Babeş-Vroman et al. ☆ | 轨迹级聚类(EM) | 轨迹内不切换 |
| 2011–14 | 非参贝叶斯(DPM-BIRL 等)☆ | 意图数不定 | 扩展性差 |
| 2022 | **DIRL**(Ashwood et al., NeurIPS 2022 ☆) | 平滑时变权重 | 与"动物离散切换策略"证据相悖;每步一次 Bellman 解 |
| 2024 | **多意图 IQL/HIQL**(Zhu et al., TMLR 2024 ★) | 一阶 Markov 链 | 无记忆;E 步 $O(nK^2)$ 前向后向 |
| 2025 | **SWIRL**(Ke et al., ICML 2025 ★) | 状态依赖转移 + 定窗历史 | 首个历史依赖策略与奖励;状态增广 $\lvert S\rvert^L$ 爆炸 |
| 2026 | **PRISM**(Sheng, Zhu & Boedecker, arXiv 2026 ★) | RNN 意图门控 | EM 精确分解为逐意图闭式子问题(IAVI 内环),$O(nK)$ E 步;首个大规模机器人数据(BridgeData V2)应用 |

应用驱动力主要来自神经科学:自由行动动物(小鼠迷宫,Rosenberg et al. 2021 ☆)的
长时程行为需要"水源—回巢—探索"这类可命名意图,单一平稳奖励解释不了。

### 8.3 风险与效用

Lazzati & Metelli (ICML 2025) ★:专家最大化 $\mathbb E[U(G)]$($G$ 为回报,$U$ 为效用)
而非 $\mathbb E[G]$——风险态度显式化为可学对象。效用同样只**部分可辨识**,
多环境示范缓解(§2.3 模式再现);CATY-UL / TRACTOR-UL 给出有限数据保证。
把"任务(奖励)"与"态度(效用/约束)"拆开表示,是 8.1 与 8.3 的共同哲学。

### 8.4 非马尔可夫奖励

SWIRL/PRISM 的历史依赖奖励之外,还有 effective-horizon 视角(arXiv:2307.06541 ◐)、
in-trajectory IRL(边观测边更新,NeurIPS 2024 ◐)等零星探索;
系统理论(何种历史依赖可辨识)尚缺(§13)。

---

## 9. 偏好时代:RLHF 即大规模 IRL

语言任务没有天然奖励,LLM post-training 的整个流水线是 IRL 思想的最大规模部署
(详细导读见 [`notes/2025_Sun_IRL_meets_LLM_Survey.md`](../notes/2025_Sun_IRL_meets_LLM_Survey.md),
综述本体为 Sun & van der Schaar, arXiv:2507.13158 ☆)。骨架三式:

1. **SFT = BC**;
2. **RLHF = 偏好反推 + soft RL**:Bradley-Terry
   $P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$ 学 $r_\phi$(Christiano et al. 2017 ☆ 起点;
   InstructGPT,Ouyang et al. 2022 ☆ 定型),再解 KL 正则目标,闭式解
   $\pi^*(y\mid x)\propto\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta}$ ——
   MaxEnt 轨迹分布,先验从均匀换成 $\pi_{\rm ref}$;
3. **DPO = 单层化**(Rafailov et al., NeurIPS 2023 ☆):反解 $r=\beta\log\frac{\pi}{\pi_{\rm ref}}+\beta\log Z(x)$
   代回 BT,$Z$ 相消——IQ-Learn 的翻版(轴 C)。

IRL 视角下的要点:

- **不可辨识性的新形状**:BT 学不到逐 prompt 平移 $c(x)$,而 $c(x)$ 恰被 $Z(x)$ 吸收——
  轨迹级奖励下"学不到的 = 不需要的";process reward(逐步打分)打破豁免,
  shaping 等价类问题(§2.2)复发。
- **Reward hacking = 可行集太大 + 分布偏移**:RM 仅在标注分布上受约束,策略主动搜索其薄弱区
  (过优化 scaling law,Gao, Schulman & Hilton, ICML 2023 ☆)。
  悲观可行集理论(§5.3)与此至今未接轨——见 §13。
- **非 BT 化**:偏好非传递/标注者异质 ⇒ 博弈化(Nash-LHF,§7)。
- **回流**:IRL 技术反哺 LLM——从专家示范学稠密推理奖励(arXiv:2510.01857 ◐)、
  failure-aware IRL 理解对齐(arXiv:2510.06092 ◐)。

---

## 10. 生成策略时代:绕开奖励的模仿

2022 年后模仿学习的工程主流悄悄换了范式:**不学 $R$,直接把策略/轨迹当生成模型学。**

- **BC 的复兴**:Diffusion Policy(Chi et al., RSS 2023 ☆)用条件扩散拟合多峰示范动作分布,
  横扫操作任务——"复合误差"没有消失,但表达力红利暂时盖过了它;
  轨迹级的 Diffuser(Janner et al., ICML 2022 ★)把规划变成条件生成。
- **接上价值函数**:离线 RL 中 Diffusion-QL(Wang et al., ICLR 2023 ★)、
  Q-Score Matching(Psenka et al., ICML 2024 ★:策略 score 对齐 $\nabla_a Q$)。
- **在线化与 Boltzmann 目标**:max-ent RL 的改进策略是 $\pi\propto e^{Q/\lambda}$
  (或镜像下降的 $\pi_{\rm old}e^{Q/\lambda}$)——**正是 MaxEnt-IRL 的孪生分布**,
  但没有目标样本可抽。2025–26 的一整条线在解决"向未归一化 Boltzmann 目标训练生成策略":
  重加权 score matching(DPMD/SDAC,Ma et al., ICML 2025 ★)、
  免似然的 PPO 代理(FPO,McAllister et al., 2025/ICLR 2026 ★;PolicyFlow,Yang et al., ICLR 2026 ★)、
  以及把噪声期望族与梯度期望族统一进后验均值估计的
  **Reverse Flow Matching**(Li, Tang & Azizan, ICML 2026 ★)。
  配套生态:离线到在线的探索注入(FINO, ICLR 2026 ★)、
  流参数化 critic(floq, ICLR 2026 ★)、后继测度的流表征(SF², ICLR 2026 ★)。
- **Flow Q-Learning**(Park, Li & Levine, ICML 2025 ★)与 guidance 理论(Feng et al., ICML 2025 ★)
  是这条线的离线支柱。

对 IRL 的意义,两面读:

> **一面**:这条线宣告"许多任务根本不需要显式奖励"——占用测度/轨迹分布视角(GAIL 的遗产)
> 借生成模型彻底兑现,轴 C 走到"连 min-max 都不要"。
> **另一面**:凡需要**迁移、外推超过示范者、推理时搜索、跨任务解释**的场景,
> 显式奖励仍是唯一的通货(AIRL 的理由从未失效;LLM 的 RM 生态即明证)。
> "要不要 $R$"的摆钟(§4.2)仍在摆。

---

## 11. 评价与基准

- **点估计时代**:EVD($V^{\pi^*_{R}}_R-V^{\pi^*_{\hat R}}_R$)与 ILE——度量学徒成功而非奖励恢复,
  两个 EVD 相同的奖励迁移行为可以截然不同。
- **可行集时代**:Hausdorff 型集合距离(Metelli 2023 ★)、reward mapping 度量(Zhao 2024 ★)、
  相容性度量(Lazzati 2024 ★)——彼此不等价,**尚无共识**;
  Zhao et al. 专门构造了"旧度量判满分、实际错误"的反例。
- **基准现状**:模仿侧被 D4RL/OGBench 等 RL 基准吸收(§10 各文皆用),
  IRL 特有的"奖励恢复质量"基准长期缺位——Arora & Doshi 2021 的抱怨在 2026 年依然成立。

---

## 12. 应用速览

- **机器人/驾驶**:MMP 一系的导航代价学习;驾驶风格与交互(Kuderer et al. 2015;Sadigh et al. 2016 ☆);
  近年重心移向 §10 的生成策略。
- **行为科学/神经科学**:轨迹预测(Ziebart 2008 出租车;Kitani et al., ECCV 2012 行人 ☆);
  自由行动动物的意图分解(§8.2:小鼠迷宫、BridgeData 机器人示范同框)。
- **对齐**:RLHF/RLAIF/DPO(§9)是 IRL 思想当前最大规模的部署;
  合作逆 RL(CIRL,Hadfield-Menell et al., NeurIPS 2016 ☆)提供人机协作的博弈框架。
- **安全**:ICRL 学隐式安全约束(§8.1);医疗决策的奖励解释(Zhao 2024 引例 ★)。

---

## 13. 开放问题

1. **可行集理论 × RLHF 工程**:悲观集值估计(§5.3)能否给 reward hacking(§9)
   一个不靠 KL 系数玄学的过优化上界?两边的度量至今对不上。
2. **函数逼近下的可行集**:线性 MDP 之外(神经 RM),$\mathcal R_{\rm feas}$
   的几何与样本复杂度几乎空白;reward compatibility(§5.4)是起点。
3. **度量共识**:§11 的三族度量哪个预测"下游任务成功"?需要一个 IRL 版的"benchmark 论文"。
4. **偏好非传递**:BT 失效时 RM 学到的对象是什么?逆博弈(§7)的 QRE 工具
   能否直接用于标注者群体?
5. **历史依赖奖励的可辨识性**:SWIRL/PRISM(§8.2)在工程上先行,
  "何种非马尔可夫结构可从行为辨识"没有定理。
6. **生成策略时代的奖励角色**:当策略是流/扩散模型,"从它反推奖励"意味着什么?
   逆问题(从流策略恢复 $Q$ 或 $R$)尚无人系统提出——floq/RFM(§10)已把正向打通,
   逆向是显然的下一步。
7. **统一的"多样性换辨识"定理**:§2.3 的模式(熵、多环境、多均衡、多次优度、多意图
   各自切等价类)散落在十几篇论文里,值得一个统一的信息论刻画。

---

## 14. 参考文献

按节分组;★/☆/◐ 含义见文首。本仓库有 PDF 的条目给相对路径。

**综述与教材**
- Arora & Doshi. *A survey of IRL: Challenges, methods and progress.* AIJ 297, 2021. [arXiv:1806.06877](https://arxiv.org/abs/1806.06877) ☆(导读:[notes](../notes/2021_Arora_Doshi_IRL_Survey.md))
- Adams, Cody & Beling. *A survey of IRL.* Artif. Intell. Rev. 55, 2022. [DOI](https://doi.org/10.1007/s10462-021-10108-x) ☆
- *Advances and applications in IRL: a comprehensive review.* Neural Comput. & Applic. 37, 2025. [DOI](https://doi.org/10.1007/s00521-025-11100-0) ◐
- Liu et al. *A Comprehensive Survey on Inverse Constrained RL.* TMLR 2025. [arXiv:2409.07569](https://arxiv.org/abs/2409.07569) ☆
- Sun & van der Schaar. *IRL Meets LLM Post-Training.* 2025. [arXiv:2507.13158](https://arxiv.org/abs/2507.13158) ☆
- Osa et al. *An Algorithmic Perspective on Imitation Learning.* FnT Robotics, 2018. ☆
- Gleave & Toyer. *A Primer on Maximum Causal Entropy IRL.* 2022. [arXiv:2203.11409](https://arxiv.org/abs/2203.11409) ☆

**§2 不适定性与可辨识**
- Ng & Russell. ICML 2000. [★PDF](../paper/基础文献/2000_Ng_Algorithms_for_IRL.pdf)
- Ng, Harada & Russell. *Policy invariance under reward transformations.* ICML 1999. ☆
- Fu, Luo & Levine. *AIRL.* ICLR 2018. [★PDF](../paper/基础文献/2018_Fu_AIRL.pdf)
- Kim, Garg, Shiragur & Ermon. *Reward Identification in IRL.* ICML 2021. [★PDF](../paper/基础文献/2021_Kim_Reward_Identification.pdf)
- Cao, Cohen & Szepesvári. *Identifiability in IRL.* NeurIPS 2021. [arXiv:2106.03498](https://arxiv.org/abs/2106.03498) ☆
- Rolland et al. *Identifiability and generalizability from multiple experts.* NeurIPS 2022. [arXiv:2209.10974](https://arxiv.org/abs/2209.10974) ☆
- Skalse et al. *Invariance in Policy Optimisation and Partial Identifiability in Reward Learning.* ICML 2023. ☆
- Kim, Deshmukh, Vlassis & Zhang. *IRL without an Optimal Demonstrator.* 2026. [arXiv:2605.30903](https://arxiv.org/abs/2605.30903) ◐
- *Closed-Form Reward Centroids.* 2025. [arXiv:2509.12010](https://arxiv.org/abs/2509.12010) ◐

**§3 经典四族**
- Abbeel & Ng. ICML 2004. [★PDF](../paper/基础文献/2004_Abbeel_Apprenticeship_Learning.pdf)
- Ratliff, Bagnell & Zinkevich. *Maximum Margin Planning.* ICML 2006. ☆
- Syed & Schapire. *A Game-Theoretic Approach to Apprenticeship Learning.* NeurIPS 2007. ☆
- Ziebart et al. AAAI 2008. [★PDF](../paper/基础文献/2008_Ziebart_MaxEnt_IRL.pdf);ICML 2010. [★PDF](../paper/基础文献/2010_Ziebart_Max_Causal_Entropy.pdf)
- Boularias, Kober & Peters. *Relative Entropy IRL.* AISTATS 2011. ☆
- Ramachandran & Amir. *Bayesian IRL.* IJCAI 2007. ☆ · Choi & Kim. *MAP-BIRL.* NeurIPS 2011. ☆ · Levine et al. *GPIRL.* NeurIPS 2011. ☆
- Klein et al. *SCIRL.* NeurIPS 2012;*CSI.* ECML 2013. ☆

**§4 深度与对抗**
- Finn, Levine & Abbeel. *GCL.* ICML 2016. [★PDF](../paper/基础文献/2016_Finn_Guided_Cost_Learning.pdf)
- Ho & Ermon. *GAIL.* NeurIPS 2016. [★PDF](../paper/基础文献/2016_Ho_GAIL.pdf)
- Ghasemipour, Zemel & Gu. *f-MAX.* CoRL 2019. ☆
- Swamy et al. *Of Moments and Matching.* ICML 2021. ☆;*IRL without RL.* ICML 2023. ☆
- Kalweit et al. *Deep Inverse Q-learning with Constraints (IAVI).* NeurIPS 2020. ☆
- Garg et al. *IQ-Learn.* NeurIPS 2021. [★PDF](../paper/基础文献/2021_Garg_IQ_Learn.pdf)
- *Trust Region IRL.* ICML 2026. ◐;arXiv:2509.23135 ◐

**§5 可行集理论**(仓库路径见 `paper/前沿文献/01`)
- Metelli, Ramponi & Restelli. ICML 2021. [★PDF](../paper/基础文献/2021_Metelli_Transferable_Rewards.pdf)
- Metelli, Lazzati & Restelli. ICML 2023. [★PDF](../paper/基础文献/2023_Metelli_Theoretical_Understanding_IRL.pdf)
- Lindner, Krause & Ramponi. *AceIRL.* NeurIPS 2022. [★PDF](../paper/前沿文献/02_Active_IRL与Environment_Design/2022_Lindner_Active_Exploration_IRL.pdf)
- Lazzati, Mutti & Metelli. *Offline IRL.* ICML 2024 ★;*Large State Spaces.* NeurIPS 2024 ★;*Reward Compatibility.* [arXiv:2501.07996](https://arxiv.org/abs/2501.07996) ◐
- Zhao, Wang & Bai. *Is IRL Harder than RL?* ICML 2024. ★
- Poiani et al. *IRL with Sub-optimal Experts.* [arXiv:2401.03857](https://arxiv.org/abs/2401.03857) ☆
- Yue et al. *CLARE.* ICLR 2023. ☆ · Brown et al. *T-REX.* ICML 2019. ☆
- Lazzati & Metelli. *Learning Utilities.* ICML 2025. ★

**§6 主动与环境设计**(`paper/前沿文献/02`)
- Kleine Büning, Villin & Dimitrakakis. ICML 2024. ★ · Bajgar et al. RLC 2025. ★
- Lopes et al. 2009;Brown et al. 2018;Kweon et al. 2023. ☆

**§7 逆博弈**(`paper/前沿文献/03`)
- Wu, Shen, Fang & Xu. NeurIPS 2022. ★ · Goktas et al. ICLR 2024. ★
- Liao et al. ICML 2025. ★ · Freihaut & Ramponi. NeurIPS 2025. ★
- Munos et al. *Nash Learning from Human Feedback.* ICML 2024. [arXiv:2312.00886](https://arxiv.org/abs/2312.00886) ☆
- Hadfield-Menell et al. *Cooperative IRL.* NeurIPS 2016. ☆

**§8 结构化**(`paper/前沿文献/06`)
- Malik et al. *ICRL.* ICML 2021. ☆ · Babeş-Vroman et al. ICML 2011. ☆
- Ashwood et al. *DIRL.* NeurIPS 2022. ☆ · Zhu et al. TMLR 2024. ★
- Ke et al. *SWIRL.* ICML 2025. ★ · Sheng, Zhu & Boedecker. *PRISM.* 2026. ★
- Rosenberg et al. eLife 2021. ☆ · Wu & Xu. *RS-MDP.* 2023. ☆

**§9 偏好与对齐**
- Christiano et al. NeurIPS 2017 ☆ · Ouyang et al. NeurIPS 2022 ☆ · Rafailov et al. NeurIPS 2023 ☆
- Gao, Schulman & Hilton. ICML 2023 ☆ · Rafailov et al. *From r to Q\*.* [arXiv:2404.12358](https://arxiv.org/abs/2404.12358) ☆
- arXiv:2510.01857 ◐ · arXiv:2510.06092 ◐

**§10 生成策略**(`paper/前沿文献/04`、`05`)
- Chi et al. *Diffusion Policy.* RSS 2023. ☆ · Janner et al. ICML 2022. ★ · Wang et al. ICLR 2023. ★
- Psenka et al. ICML 2024. ★ · Ma et al. ICML 2025. ★ · Park, Li & Levine. ICML 2025. ★ · Feng et al. ICML 2025. ★
- McAllister et al. *FPO.* ICLR 2026. ★ · Yang et al. *PolicyFlow.* ICLR 2026. ★ · Li, Tang & Azizan. *RFM.* ICML 2026. ★
- Shin et al. *FINO.* ICLR 2026. ★ · Agrawalla et al. *floq.* ICLR 2026. ★ · Shi et al. *SF².* ICLR 2026. ★

**§11–12**
- Ross & Bagnell. AISTATS 2010/2011(复合误差、DAgger)☆ · Kitani et al. ECCV 2012 ☆
- Kuderer et al. ICRA 2015 ☆ · Sadigh et al. RSS 2016 ☆ · *In-Trajectory IRL.* NeurIPS 2024 ◐

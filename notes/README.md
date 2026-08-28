# 五篇论文的一条线：从“反推 reward”到“学一个轨迹分布”

已读的五篇，按它们**互相接手**的顺序整理。
每一节的结构都是：*上一篇留下什么缺口 → 这篇怎么补 → 它自己又留下什么*。

| # | 论文 | 总结 | 链接 |
|---|---|---|---|
| 1 | Ng & Russell, ICML 2000 | 写下“专家最优”的线性不等式，发现解不唯一 | [ACM DL](https://dl.acm.org/doi/10.5555/645529.657801) |
| 2 | Abbeel & Ng, ICML 2004 | 不解 reward，改匹配 feature expectation | [Stanford 项目页](http://ai.stanford.edu/~pabbeel/irl/) |
| 3 | Ziebart et al., AAAI 2008 | 在满足匹配的**分布**里取最大熵 | [AAAI PDF](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf) |
| 4 | Ziebart, Bagnell & Dey, ICML 2010 | 熵换成因果熵，修正随机环境下的偏差 | [ACM DL](https://dl.acm.org/doi/10.5555/3104322.3104481) |
| 5 | Finn, Levine & Abbeel, ICML 2016 | 用采样近似 $Z_\theta$，脱离 tabular | [arXiv:1603.00448](https://arxiv.org/abs/1603.00448) · [PMLR](https://proceedings.mlr.press/v48/finn16.html) |

---

## 0. 统一符号

五篇的记号各不相同，先对齐，后面才不会乱：

| 概念 | 本文用 | 各篇原本写成 |
|---|---|---|
| MDP | $M=(S,A,P,\gamma,R)$ | 一致 |
| 转移矩阵 | $P_a\in\mathbb R^{\lvert S\rvert\times\lvert S\rvert}$，$(P_a)_{ss'}=P(s'\mid s,a)$ | Ng & Russell 用 $\mathbf P_a$ |
| 特征 | $\phi(s)\in\mathbb R^k$ | Abbeel & Ng 用 $\phi$，Ziebart 用 $\mathbf f_s$ |
| 轨迹 | $\tau=(s_0,a_0,s_1,\dots)$ | 一致 |
| 轨迹特征计数 | $f(\tau)=\sum_t\gamma^t\phi(s_t)$ | Ziebart 用 $\mathbf f_\tau$。**注意**：2004 带折扣 $\gamma^t$；2008 是有限视界**无折扣**的 $\sum_{s_t\in\tau}\phi(s_t)$（论文 eq. 1），下文 §3 用的即是后者 |
| 特征期望 | $\mu(\pi)=\mathbb E_\pi[f(\tau)]$ | Abbeel & Ng 的核心量 |
| 示范的经验特征 | $\tilde f=\frac1m\sum_{i=1}^m f(\tau_i)$ | Ziebart 用 $\tilde{\mathbf f}$ |
| reward 参数 | $\theta$ | 2000 直接用 $R$；2004 用 $w$；2016 用 **cost** $c_\theta$（符号相反） |

**注意第 5 篇的符号翻转**：前四篇最大化 reward，Finn 最小化 cost，
所以那里的指数是 $\exp(-c_\theta)$ 而不是 $\exp(+\theta^\top f)$。看公式时要留意正负号。

---

## 1. Ng & Russell (2000)：得到的是一个锥，不是一个解

**问题设定**：已知 MDP 的 $P$ 和 $\gamma$，已知专家策略 $\pi$（**在每一个状态上**），求 $R$。

### 1.1 最优性条件

不失一般性设 $\pi(s)\equiv a_1$。由 Bellman 方程

$$V^\pi = R + \gamma P_{a_1}V^\pi \quad\Longrightarrow\quad V^\pi=(I-\gamma P_{a_1})^{-1}R$$

而 $Q^\pi(s,a)=R(s)+\gamma\,(P_aV^\pi)(s)$。$\pi$ 最优的充要条件是对所有 $s$ 与 $a\neq a_1$：

$$Q^\pi(s,a_1)\ \ge\ Q^\pi(s,a)
\iff \gamma(P_{a_1}V^\pi)(s)\ \ge\ \gamma(P_aV^\pi)(s)$$

两边的 $R(s)$ 抵消、$\gamma>0$ 除掉，得到论文的核心不等式：

$$\boxed{\ (P_{a_1}-P_a)\,(I-\gamma P_{a_1})^{-1}\,R\ \succeq\ 0\quad \forall a\in A\setminus\{a_1\}\ }\tag{$\star$}$$

$R$ 在 $(\star)$ 里是**线性且齐次**的。三个直接后果：

1. $R\equiv 0$ 满足每一条不等式——所有策略都最优，所以 $\pi$ 当然最优。**退化解**。
2. $R$ 是解 $\Rightarrow$ $cR$（$c>0$）也是解；解集对非负线性组合封闭。
   所以 $(\star)$ 的解集是一个**多面锥**。
3. 于是 $(\star)$ 给出的是**可行域**，不是解。要拿到单一个 $R$，必须外加准则。

### 1.2 论文加的准则

$$\max_{R}\ \sum_{s\in S}\ \min_{a\in A\setminus\{a_1\}}\Big\{\big[(P_{a_1}-P_a)(I-\gamma P_{a_1})^{-1}R\big](s)\Big\}\ -\ \lambda\lVert R\rVert_1$$

$$\text{s.t.}\quad (\star),\qquad \lvert R(s)\rvert\le R_{\max}$$

- 第一项：把专家动作与**次优**动作的 $Q$ 差拉到最大 → 间隔最大化。
- 第二项：偏好稀疏 / 简单的 reward。

这两项都是**选择**，不是从问题推导出来的。换一个正则项就换一个答案。
论文另外给了线性函数近似版 $R(s)=\sum_i\alpha_i\varphi_i(s)$，
以及只用采样轨迹（不需要完整 $\pi$）的第三个算法——那是后面所有工作的种子。

### 1.3 留下的缺口

- **(a)** 需要**完整的** $\pi$。现实只有几条轨迹，没走过的状态完全不受约束。
- **(b)** 假设专家严格最优。专家次优一点点，$(\star)$ 可能**整组无解**（LP infeasible）。
- **(c)** 从锥里挑哪一点是任意的。

→ (a) 由第 2 篇处理，(b)(c) 由第 3 篇处理。

---

## 2. Abbeel & Ng (2004)：把目标从 reward 换成 $\mu(\pi)$

**接手 (a)**：只用轨迹，不需要完整策略。
**绕开 (c)**：既然 reward 挑不准，那就别挑。

### 2.1 关键的一步：$V$ 对 $\mu$ 是线性的

假设 reward 对特征线性：$R(s)=w^\top\phi(s)$，$\lVert w\rVert_2\le1$，$\phi:S\to[0,1]^k$。则

$$V^\pi(w)=\mathbb E\Big[\sum_{t=0}^\infty\gamma^t\,w^\top\phi(s_t)\ \Big|\ \pi\Big]
= w^\top\underbrace{\mathbb E\Big[\sum_{t=0}^\infty\gamma^t\phi(s_t)\ \Big|\ \pi\Big]}_{\displaystyle \mu(\pi)\ \in\ \mathbb R^k}$$

于是对**任意** $\lVert w\rVert_2\le1$：

$$\big\lvert V^{\tilde\pi}(w)-V^{\pi_E}(w)\big\rvert
=\big\lvert w^\top(\mu(\tilde\pi)-\mu_E)\big\rvert
\ \le\ \lVert w\rVert_2\,\lVert\mu(\tilde\pi)-\mu_E\rVert_2
\ \le\ \lVert\mu(\tilde\pi)-\mu_E\rVert_2$$

**右边没有 $w$。** 这一步把“我不知道真实 reward”整个绕开了：
只要 $\lVert\mu(\tilde\pi)-\mu_E\rVert_2\le\epsilon$，
$\tilde\pi$ 在**所有**线性 reward 下都不比专家差过 $\epsilon$。

目标于是从“recover $R$”换成“match $\mu$”。

经验估计（$m$ 条示范）：

$$\hat\mu_E=\frac1m\sum_{i=1}^m\sum_{t=0}^{H}\gamma^t\phi\big(s_t^{(i)}\big)$$

样本复杂度（Thm 2，论文 eq. 19 的精确形式）：

$$m\ \ge\ \frac{2k}{(\epsilon(1-\gamma))^2}\,\log\frac{2k}{\delta}$$

（对每个座标用 Hoeffding：$(1-\gamma)\hat\mu_i\in[0,1]$，再对 $k$ 个座标 union bound。）
**依赖 $k$ 而不是 $\lvert S\rvert$**——
这是它能离开 tabular 的原因。

### 2.2 max-margin 版：$w$ 就是分离方向

第 $i$ 轮解

$$t^{(i)}=\max_{\lVert w\rVert_2\le1}\ \min_{j\in\{0,\dots,i-1\}}\ w^\top\big(\mu_E-\mu^{(j)}\big)$$

这是一个标准 SVM：$\mu_E$ 当正例、已经产生过的 $\{\mu^{(j)}\}$ 当负例，
$w^{(i)}$ 是**分离超平面的法向量**，$t^{(i)}$ 是**间隔**。
然后用 $R=(w^{(i)})^\top\phi$ 解一次 MDP 得到 $\pi^{(i)}$，算出 $\mu^{(i)}=\mu(\pi^{(i)})$，进入下一轮。

若 $t^{(i)}\le\epsilon$ 就停：此时**没有任何** $\lVert w\rVert\le1$ 能让专家比目前这组策略好过 $\epsilon$。

### 2.3 projection 版：奖励权重 = 残差方向

不需要 QP solver 的等价写法。令 $\bar\mu^{(0)}=\mu^{(0)}$，第 $i$ 轮：

$$\bar\mu^{(i)}=\bar\mu^{(i-1)}
+\frac{\big(\mu^{(i)}-\bar\mu^{(i-1)}\big)^\top\big(\mu_E-\bar\mu^{(i-1)}\big)}
{\big(\mu^{(i)}-\bar\mu^{(i-1)}\big)^\top\big(\mu^{(i)}-\bar\mu^{(i-1)}\big)}
\Big(\mu^{(i)}-\bar\mu^{(i-1)}\Big)$$

$$w^{(i+1)}=\mu_E-\bar\mu^{(i)},\qquad t^{(i+1)}=\big\lVert\mu_E-\bar\mu^{(i)}\big\rVert_2$$

第一式就是把 $\mu_E$ 正交投影到线段 $\overline{\bar\mu^{(i-1)}\,\mu^{(i)}}$ 上。
几何上 $\bar\mu^{(i)}$ 逐步逼近 $\mu_E$ 在 $\operatorname{conv}\{\mu^{(j)}\}$ 上的投影，而

> **$w$ 是从凸包指向 $\mu_E$ 的残差向量。**
> “reward 权重”和“分离方向”是同一个东西——这是这篇最漂亮的一点。

残差方向指出“专家在哪个特征维度上比目前所有策略都强”，
把它当 reward 去解 MDP，就是去补上那个维度。这是 Frank–Wolfe 式的贪婪逼近，
所以间隔以 $t^{(i)}=O(1/\sqrt i)$ 的速率下降，
迭代次数 $n=O\!\big(\tfrac{k}{(1-\gamma)^2\epsilon^2}\log\tfrac{k}{(1-\gamma)\epsilon}\big)$。

已对照论文核实（2026-08-28，对照 `paper/基础文献/2004_Abbeel_Apprenticeship_Learning.pdf`）：

$$\text{Thm 1:}\quad n=O\!\left(\frac{k}{(1-\gamma)^2\epsilon^2}\,\log\frac{k}{(1-\gamma)\epsilon}\right)
\qquad\text{Thm 2:}\quad m\ \ge\ \frac{2k}{(\epsilon(1-\gamma))^2}\,\log\frac{2k}{\delta}$$

论文还给了一条容错结果：若真实 reward 不在 $\phi$ 的张成里、残差为 $\varepsilon(s)$，
性能退化以 $\lVert\varepsilon\rVert$ 的量级优雅衰减（graceful degradation），
这正是 §2.5 缺口 (d) 的定量版本。

### 2.4 为什么输出是混合策略

$\mu_E$ 一般**不等于**任何单一 $\mu(\pi^{(j)})$，但会落在 $\operatorname{conv}\{\mu^{(j)}\}$ 里。
而 $\mu$ 对“策略的混合”是线性的：

$$\mu\Big(\textstyle\sum_j\lambda_j\pi^{(j)}\Big)=\sum_j\lambda_j\,\mu\big(\pi^{(j)}\big)$$

所以算法返回的是一组 $\{\lambda_j\}$：**开局掷一次骰子选出 $\pi^{(j)}$，然后整局照它走。**
注意这是“每局随机”，不是“每步随机”——实际中很尴尬。

### 2.5 留下的缺口

- **(d)** 只保证“表现一样好”，**拿不到** reward。想转移到新环境就没东西可转移。
- **(e)** 混合策略的形式不自然。
- **(f)** 满足 $\mu$ 匹配的策略**依然有无穷多个**（不同策略可以有相同的 $\mu$）。该挑哪一个？

→ (f) 正是第 3 篇的入口。

---

## 3. Ziebart et al. (2008)：从“挑一个策略”到“学一个分布”

**接手 (f)**，顺带处理 (b)（专家次优）与 (c)（挑选准则的任意性）。

### 3.1 最大熵原理

不再问“挑哪个策略”，改问“**挑哪个轨迹分布**”。在特征匹配的约束下求熵最大：

$$\max_{P}\ \ H(P)=-\sum_\tau P(\tau)\log P(\tau)$$

$$\text{s.t.}\quad \sum_\tau P(\tau)f(\tau)=\tilde f,\qquad \sum_\tau P(\tau)=1,\qquad P(\tau)\ge0$$

Lagrangian（$\theta$ 对应特征约束，$\eta$ 对应归一化）：

$$\mathcal L=-\sum_\tau P\log P+\theta^\top\Big(\sum_\tau P\,f(\tau)-\tilde f\Big)+\eta\Big(\sum_\tau P-1\Big)$$

$$\frac{\partial\mathcal L}{\partial P(\tau)}=-\log P(\tau)-1+\theta^\top f(\tau)+\eta=0$$

$$\Longrightarrow\quad\boxed{\ P(\tau\mid\theta)=\frac{1}{Z(\theta)}\exp\big(\theta^\top f(\tau)\big),\qquad
Z(\theta)=\sum_\tau\exp\big(\theta^\top f(\tau)\big)\ }$$

**指数族。** 而 $\theta^\top f(\tau)=\sum_t\theta^\top\phi(s_t)$ 就是这条轨迹的总 reward，所以模型在说：

> reward 高的轨迹**指数级**地更可能被走，但不是绝对地被走。

专家次优不再让问题无解，只是让那条轨迹的似然低一点——缺口 (b) 自动消失。

### 3.2 对偶：最大熵 = 极大似然

约束最大熵问题的对偶，恰好是在这个指数族上做 MLE：

$$L(\theta)=\frac1m\sum_{\tau_i\in D}\log P(\tau_i\mid\theta)=\theta^\top\tilde f-\log Z(\theta)$$

$$\nabla_\theta L=\tilde f-\frac{\nabla_\theta Z(\theta)}{Z(\theta)}
=\tilde f-\sum_\tau P(\tau\mid\theta)f(\tau)
=\tilde f-\mathbb E_{P_\theta}\big[f(\tau)\big]$$

写成状态层级（$f(\tau)=\sum_t\phi(s_t)$）：

$$\boxed{\ \nabla_\theta L=\tilde f-\sum_{s}D_s\,\phi(s)\ }$$

其中 $D_s$ 是**期望状态访问频率**（expected state visitation frequency）。

“**观察值 − 模型期望值**”——所有指数族模型共通的梯度形式。

二阶导：

$$\nabla^2_\theta L=-\operatorname{Cov}_{P_\theta}\big[f(\tau)\big]\ \preceq\ 0$$

所以 $L$ 是**凹的**，梯度上升收敛到全局最优。缺口 (c) 也消失了：
解唯一，而且“最大熵”这个准则有原理上的理由（不引入示范没告诉你的偏好），
不像 2000 的 $\ell_1$ 那样纯属选择。

### 3.3 怎么算 $D_s$（Algorithm 1）

贵的是 $\mathbb E_{P_\theta}[f]$。论文用一组 backward–forward 递推：

**backward（算局部 partition）**。论文原文：初始化 $Z_{s}=1$（对所有状态），迭代 $N$ 次

$$Z_{s,a}\ \leftarrow\ \sum_{s'}P(s'\mid s,a)\ e^{\theta^\top\phi(s)}\ Z_{s'},
\qquad Z_s\ \leftarrow\ \sum_a Z_{s,a}$$

（很多课程讲义写成“只在 terminal 状态置 1”的吸收态变体；论文本身是全 1 初始化 + 固定 $N$ 次迭代，
两者在有吸收目标、视界足够长时一致。）

**局部动作概率**：

$$\pi_\theta(a\mid s)=\frac{Z_{s,a}}{Z_s}$$

**forward（算访问频率）**：

$$D_{s,0}=d_0(s),\qquad
D_{s',t+1}=\sum_{s}\sum_a D_{s,t}\ \pi_\theta(a\mid s)\ P(s'\mid s,a),\qquad
D_s=\sum_{t=0}^{H}D_{s,t}$$

> ⚠️ **论文原文的 forward pass 印刷有误**（AAAI 2008 版 Algorithm 1 第 5 步）：
> $D_{s_i,t+1}=\sum_{a_{i,j}}\sum_k D_{s_k,t}P(a_{i,j}\mid s_i)P(s_k\mid a_{i,j},s_i)$ ——
> 下标 $i,k$ 自相矛盾（右边是从 $s_i$ 出发的转移，却用来更新 $s_i$ 自己）。
> 上面写的是守恒的正确版本；实现时用 $\sum_s D_{s,t}=1$（每步）或 $\sum_s D_s=H$ 当检查，
> 这条守恒律恰好就能抓出这类下标错误。

### 3.4 概念上的跃迁

这篇真正的贡献不是算法，是**问题的重新表述**：

> IRL 从“找一个 reward / 一个策略”变成“**学一个轨迹上的概率分布**”，
> $\theta$ 是这个分布的自然参数。

一旦写成指数族，整套概率建模的工具（MLE、凸性、对偶、图模型推论）就全部可以用了。
后面三篇都建立在这个表述上。

### 3.5 留下的缺口

- **(g)** $Z(\theta)=\sum_\tau(\cdot)$ 要遍历所有轨迹 → 只在 tabular + 有限 horizon 下算得动。
- **(h)** **随机 dynamics 下这个模型是错的。** → 第 4 篇。

---

## 4. Ziebart, Bagnell & Dey (2010)：因果熵

**接手 (h)。**

### 4.1 错在哪

$P(\tau)\propto\exp(\theta^\top f(\tau))$ 对**整条轨迹**赋概率。
但轨迹里有两种东西：agent 选的动作 $a_t$，以及环境掷骰子决定的转移结果 $s_{t+1}$。
把两者一起丢进 $\exp$ 里归一化，等于让模型**替 agent 挑选环境的随机结果**——
agent 被赋予了它并不具有的控制力，于是模型系统性地**高估**了 agent 对结果的掌控（乐观 / risk-seeking 偏差）。

看 backup 最清楚。轨迹级最大熵在随机 dynamics 下导出的递推是

$$V(s)=\log\sum_a\exp\Big(\theta^\top\phi(s)+\underbrace{\log\ \mathbb E_{s'\sim P(\cdot\mid s,a)}\big[e^{V(s')}\big]}_{\text{soft-max over } s'}\Big)$$

里面那个 $\log\mathbb E[e^{V}]$ 会把**运气好的后继状态**加权得过高。

### 4.2 因果熵

修法：限制策略只能依赖**已经发生**的信息。定义因果条件概率

$$P\big(a_{1:T}\,\big\|\,s_{1:T}\big):=\prod_{t=1}^{T}P\big(a_t\ \big|\ s_{1:t},\,a_{1:t-1}\big)$$

注意 $a_t$ 只条件在 $s_{1:t}$ 上，**看不到未来的** $s_{t+1:T}$。对应的因果熵：

$$H\big(A^T\,\big\|\,S^T\big)=\mathbb E\Big[-\log P\big(A^T\big\|S^T\big)\Big]
=\sum_{t=1}^{T}H\big(A_t\ \big|\ S_{1:t},A_{1:t-1}\big)$$

最大化因果熵、约束仍是特征匹配，且 dynamics $P$ 固定不可优化。解得 **soft Bellman recursion**：

$$Q^{\text{soft}}_\theta(s,a)=\theta^\top\phi(s,a)+\mathbb E_{s'\sim P(\cdot\mid s,a)}\big[V^{\text{soft}}_\theta(s')\big]$$

$$V^{\text{soft}}_\theta(s)=\operatorname*{softmax}_a Q^{\text{soft}}_\theta(s,a)=\log\sum_a\exp Q^{\text{soft}}_\theta(s,a)$$

$$\pi_\theta(a\mid s)=\exp\Big(Q^{\text{soft}}_\theta(s,a)-V^{\text{soft}}_\theta(s)\Big)$$

### 4.3 整篇的差别就是这张表

| | 对动作 $a$ | 对后继状态 $s'$ |
|---|---|---|
| 2008（轨迹级最大熵） | $\log\sum_a\exp(\cdot)$ | $\log\mathbb E_{s'}\big[e^{(\cdot)}\big]$ ← **错** |
| 2010（因果熵） | $\log\sum_a\exp(\cdot)$ | $\mathbb E_{s'}\big[(\cdot)\big]$ ← **对** |

动作上仍然是 softmax（agent 确实在选动作），
但转移上必须是**普通期望**（agent 不能选环境）。

**确定性 dynamics 下两者重合**——所以 2008 在确定性环境里是对的，
它的问题只在随机环境浮现。

梯度形式完全没变：

$$\nabla_\theta L=\tilde f-\mathbb E_{\pi_\theta,\,P}\big[f\big]$$

右边改用 $\pi_\theta$ 配**真实** $P$ 做前向传播算出来。

### 4.4 留下的缺口

- **(i)** 仍然是 tabular + 已知 dynamics + 线性 reward。三个限制一个都没松。

---

## 5. Finn, Levine & Abbeel (2016)：采样近似 $Z_\theta$

**接手 (i) 的全部三条。**

### 5.1 设定（注意符号翻转为 cost）

$c_\theta(\tau)$ 是神经网络，$\tau$ 在连续高维空间。能量模型：

$$p_\theta(\tau)=\frac{1}{Z_\theta}\exp\big(-c_\theta(\tau)\big),\qquad
Z_\theta=\int\exp\big(-c_\theta(\tau)\big)\,\mathrm d\tau$$

负对数似然：

$$\mathcal L(\theta)=-\frac1N\sum_{\tau_i\in\mathcal D_{\text{demo}}}\log p_\theta(\tau_i)
=\frac1N\sum_{i}c_\theta(\tau_i)+\log Z_\theta$$

$$\nabla_\theta\mathcal L=\underbrace{\frac1N\sum_i\nabla_\theta c_\theta(\tau_i)}_{\text{demo}}
-\underbrace{\mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta c_\theta(\tau)\big]}_{\text{model}}$$

**还是同一个梯度**（示范 − 模型期望），只是 reward 换成 cost 所以正负号翻了。
问题完全集中在第二项：$p_\theta$ 采样不了，$Z_\theta$ 是高维积分。

### 5.2 importance sampling

引入一个**能采样**的分布 $q(\tau)$：

$$Z_\theta=\int q(\tau)\,\frac{e^{-c_\theta(\tau)}}{q(\tau)}\,\mathrm d\tau
\ \approx\ \frac1M\sum_{\tau_j\sim q}w_j,
\qquad w_j:=\frac{e^{-c_\theta(\tau_j)}}{q(\tau_j)}$$

模型期望项用自归一化的 IS 估计：

$$\mathbb E_{p_\theta}\big[\nabla_\theta c_\theta\big]\ \approx\
\frac{\sum_j w_j\,\nabla_\theta c_\theta(\tau_j)}{\sum_j w_j}$$

### 5.3 $q$ 该选什么——关键的那一步

IS 的方差在 $q\propto e^{-c_\theta}$（也就是 $q=p_\theta$）时最小。而

$$\operatorname{KL}\big(q\,\|\,p_\theta\big)
=\mathbb E_q\big[c_\theta(\tau)\big]-\mathcal H(q)+\log Z_\theta$$

$\log Z_\theta$ 与 $q$ 无关，所以“让 $q$ 靠近 $p_\theta$”等价于

$$\min_q\ \ \mathbb E_q\big[c_\theta(\tau)\big]-\mathcal H(q)$$

**这恰好是一个最大熵 RL 问题**（最小化 cost + 最大化熵）。于是：

> “近似 partition function”与“用当前 cost 训练一个策略”是**同一件事**。

这就是 guided cost learning 的名字由来，也是它跳出 tabular 的方式：
不再枚举轨迹，而是**训练一个策略去产生轨迹**。

### 5.4 算法

交替两步：

1. 用当前的 $c_\theta$ 跑若干步 MaxEnt policy optimization，更新 $q$；
2. 从 $q$ 采样，配合示范，用 5.2 的 IS 梯度更新 $\theta$。

**实现上的两个要点**：

- proposal 用**混合分布**而不是纯 $q$：

  $$\mu(\tau)=\tfrac12\,q(\tau)+\tfrac12\,\hat p_{\text{demo}}(\tau),
  \qquad w_j=\frac{e^{-c_\theta(\tau_j)}}{\mu(\tau_j)}$$

  因为训练早期的 $q$ 几乎覆盖不到示范所在的区域，纯 $q$ 会让权重爆炸。

- cost 是神经网络 → 必须加正则，否则会退化成无意义的解。论文 §5 的两个正则项（对轨迹 $\tau$ 上的状态 $x_t$）：

  $$g_{\text{lcr}}(\tau)=\sum_{x_t\in\tau}\big[(c_\theta(x_{t+1})-c_\theta(x_t))-(c_\theta(x_t)-c_\theta(x_{t-1}))\big]^2$$

  惩罚 cost 的二阶时间差分（“局部匀速”），通用；

  $$g_{\text{mono}}(\tau)=\sum_{x_t\in\tau}\big[\max\big(0,\ c_\theta(x_t)-c_\theta(x_{t-1})-1\big)\big]^2$$

  squared hinge，要求示范的 cost 随时间单调下降，只适用于“到达目标”类的 episodic 任务。

### 5.5 留下的缺口 / 接下来

- $q$ 与 $c_\theta$ 的交替，形式上就是 **generator vs. discriminator** → GAIL / AIRL（还没读）。
- 三个限制松开了，但 **reward 的可辨识性问题一点都没碰**——
  这条线从 2000 到 2016 一直悬著。

---

## 6. 贯穿五篇的那一个梯度

从 2008 开始，三篇的梯度**完全是同一条**：

$$\boxed{\ \nabla_\theta=\underbrace{\tilde f}_{\text{demo}}\ -\ \underbrace{\mathbb E_{\text{model}(\theta)}\big[f\big]}_{\text{model}(\theta)}\ }$$

变的只有**右边那一项怎么算**：

| 论文 | 模型期望项的算法 | 适用范围 |
|---|---|---|
| Ziebart 2008 | backward–forward 动态规划 | tabular，确定性 dynamics |
| Ziebart 2010 | soft Bellman + 前向传播 | tabular，**随机** dynamics |
| Finn 2016 | importance sampling，proposal 由 MaxEnt RL 训练 | 连续高维，**未知** dynamics |

而 2000 与 2004 是这条梯度的前史：
2000 给出**可行域**（但没有目标函数），
2004 给出**该匹配什么**（$\mu$，但只当成几何逼近），
2008 才把“匹配 $\mu$”变成一个概率模型的 MLE，于是有了梯度。

## 7. 五层递进，各一句话

| 层 | 论文 | 这一层做的事 |
|---|---|---|
| 1 | Ng & Russell 2000 | 写下“专家最优”的线性不等式 $(\star)$ → 得到一个**锥**，不是一个解 |
| 2 | Abbeel & Ng 2004 | 不解 reward，改让 $\mu(\pi)$ 对上 → **绕开**不唯一性，代价是只得到混合策略、拿不到 reward |
| 3 | Ziebart 2008 | 在满足 $\mu$ 匹配的**分布**里取最大熵 → 唯一解，且是指数族 MLE，凸问题 |
| 4 | Ziebart 2010 | 熵 → 因果熵 → 修正随机环境下的乐观偏差，得到 soft Bellman |
| 5 | Finn 2016 | $Z_\theta$ 用 IS 近似，proposal 用 MaxEnt RL 训练 → 脱离 tabular / 已知 dynamics / 线性 reward |

## 8. 待办

**回去核对**（2026-08-28 已对照 `paper/基础文献/` 的 PDF 核完）

- [x] Abbeel & Ng 的两个 bound 确切常数 → 已补进 §2.3
- [x] Finn 2016 的正则项（lcr、单调性）具体形式 → 已补进 §5.4
- [x] Ziebart 2010 的证明结构：原文 Thm（eq. 5–6）是“凸原问题 + 拉格朗日 + 强对偶”，
      对 $P(A\|S)$ 求导置零得 $P_\theta(A_t\mid S_t)\propto\exp\{\theta^\top\mathbb E[F]-\sum_{\tau>t}\mathbb E[\log P_\theta]\}$，
      再代入递归式验证。完整的逐步推导仍值得自己手写一遍：
- [ ] 手推 Ziebart 2010 eq. 4 → eq. 5 的完整拉格朗日（读懂证明骨架 ≠ 自己推得出来）
- [ ] 发现的新问题：AAAI 2008 原文 Algorithm 1 forward pass 的下标印刷有误（见 §3.3）——
      去查一下 Ziebart 2010 博士论文里的版本是否已修正

**五篇留下的、还没读到答案的**

- [ ] reward 的不可辨识性：五篇全都绕开了它，没有一篇正面处理
      → Ng, Harada & Russell (1999) 的 potential-based shaping 是起点
- [ ] $q$ vs. $c_\theta$ 的交替和 GAN 的关系 → GAIL / AIRL
- [ ] 多方情境：这五篇都是单 agent，接到 inverse game theory 要换什么

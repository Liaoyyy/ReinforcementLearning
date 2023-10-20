# 机器学习基础

## 损失函数
交叉熵 \
对两个m维离散分布
$$
\boldsymbol {p}=[p_1,...,p_m]^T  \quad \boldsymbol q = [q_1,...,q_m]^T
$$
交叉熵：
$$
H(\boldsymbol{p,q})=-\sum_{j=1}^m p_j \cdot lnq_j
$$
KL散度：
$$
\begin{align}
KL(\boldsymbol{p,q})&=\sum_{j=1}^m p_j \cdot ln \frac{p_j}{q_j} \nonumber \\
&=H(\boldsymbol{p,q})-H(\boldsymbol p) \nonumber
\end{align}
$$
KL表征两个两个概率分布区别大小，当p、q同分布时KL=0

## 蒙特卡洛模拟
案例:  
求近似期望
$$
E_{X \sim p(\cdot)} [f(X)] = \int_{\Omega} p(x) \cdot f(x) dx
$$
抽取n个样本，记为向量$x_1,...,x_n \sim p(\cdot)$，可近似求取
$$
q_n=\frac{1}{n} \sum_{i=1}^n f(x_i)
$$
等价于
$$
q_n=(1-\frac{1}{n}) \cdot q_{n-1} + \frac{1}{n} \cdot f(x_t)
$$
替换$\frac{1}{n}$为$\alpha_n$
$$
q_n=(1-\alpha_n) \cdot q_{n-1} + \alpha_n \cdot f(x_t)
$$
以上公式为**Robbins_Monro**算法 \
$\alpha_t$需满足
$$
\lim_{n \to \infty} \sum_{t=1}^n \alpha_t = \infty \\
\lim_{n \to \infty} \sum_{t=1}^n \alpha_t^2 < \infty
$$

## 强化学习基本概念
### 马尔科夫决策过程(MDP)
**状态**：环境在当前时刻的状况 \
**状态空间**：所有可能存在状态的集合，记为$S$ \
**动作**：智能体基于当前状态做出的决策 \
**动作空间**：所有可能动作集合 \
**奖励**：智能体执行一个动作后，环境反馈给智能体一个数值 \
**状态转移**：智能体从当前时刻的状态$s$转移到下一个时刻状态为$s'$的过程 \
**状态转移概率函数**：状态转移具有随机性
$$
p_t(s'|s,a) = P(S'_{t+1}=s'|S_t=s,A_t=a)
$$
表示在当前状态$s$，智能体执行动作$a$，环境状态变为$s'$

### 随机性来源
1. 策略函数
   给定当前状态$s$，策略函数$\pi(a|s)$会算出动作空间中每个动作$a$概率值
2. 状态转移函数
   当状态$s$和$a$确定时，下一个状态仍有随机性。用状态转移函数$p(s'|s,a)$计算所有可能状态的概率

马尔科夫性质：
$$
\mathbb{P}(S_{t+1}|S_t,A_t)=\mathbb{P}(S_{t+1}|S_1,A_1,S_2,A_2,...,S_t,A_t)
$$
即下一时刻状态仅依赖于当前状态$S_t$和动作$A_t$，而不依赖于过去状态和动作

### 回报和折扣回报
1. **回报**：当前时刻开始到本回合结束所有奖励总和（智能体**目标为让回报尽量大，而非最大化当前奖励**）
2. **折扣汇报**：在MDP中，给未来的奖励做折扣，即$U_t = R_t+\gamma \cdot R_{t+1}+ \gamma^2 \cdot R_{t+2}+ \gamma^3 \cdot R_{t+3}+ ...$ \
   此时，$\gamma \in [0,1]$

有限期：MDP存在终止状态 \
无限期：不存在终止状态，$\gamma=1$时回报等于无穷

### 价值函数
1. 动作价值函数
   $$
   Q_\pi(s_t,a_t)=\mathbb{E}_{S_{t+1},A_{t+1},...,S_n,A_n}[U_t|S_t=s_t,A_t=a_t]
   $$
   表示在当前时刻（已观测到$S_t$与$A_t$的值），对未来回报的期望
2. 最优动作价值函数（用于排除策略$\pi$影响，只评价当前状态和动作好坏）
   $$
   Q_{\star}(s_t,a_t)=\max_{\pi} Q_{\pi}(s_t,a_t)
   $$
   即在多种策略函数下选择最优策略函数
3. 状态价值函数（衡量当前状态是否有利）
   $$
   V_{\pi}(s_t)=\mathbb{E}_{A_t\sim \pi(\cdot|s_t)}[Q_{\pi}(s_t,a_t)]
   $$
   即对动作$A_t$求期望
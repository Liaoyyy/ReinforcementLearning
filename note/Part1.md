# 表格型求解方法

## ch2（非关联性任务）
### 动作选择规则
1. 贪心方法：每次选择具有最高估计值的动作
2. $\epsilon$-贪心方法：以$\epsilon$（很小）概率从所有动作中等概率随机做出选择，以$1-\epsilon$的概率选择最高估计值动作 \
两种方法对比：方法选择取决于任务情况--当收益噪声较大时，选择$\epsilon$-贪心方法可做更多尝试找到实际最优值；当噪声小时，贪心方法可更快找到最佳动作
3. 基于置信度上界(upper confidence bound, UCB)的动作选择方法
   $$A_t=argmax_a\left[Q_t(a)+c\sqrt\frac{\ln{t}}{N_t(a)}\right]$$
   其中$N_t(a)$表示时刻$t$之前动作a被选择的次数 \
   *优点*：$\epsilon$-贪心方法是一种盲目的选择，而该算法可尝试被选择次数少且价值估计较高的动作（更有潜力）\
   *缺点*：处理非平稳问题和较大状态空间存在困难

**平稳问题**：收益概率分布不随时间变化 \
对非平稳问题，$\epsilon$-贪心方法更有效

### 动作价值的估计
1. 平均值估计(**常用于平稳问题**)
   $$
   \begin{align}
   Q_{n+1}&=Q_n+\frac{1}{n}\left[R_n-Q_n\right] \nonumber\\
   &=\frac{R_1+R_2+...+R_{n}}{n} \nonumber
   \end{align}
   $$
   其中$Q_n$为一个动作被选择$n-1$次后它的估计的动作价值,$R_n$为这一动作被选择第n次后获得的收益\ 
   此时等价于$n$次尝试后的动作价值取平均值
2. 指数近因加权估计(**常用于非平稳问题**)
   $$
   \begin{align}
   Q_{n+1}&=Q_n+\alpha\left[R_n-Q_n\right] \nonumber\\
   &=\alpha R_n+(1-\alpha) Q_n \nonumber
   \end{align}
   $$
   其中步长$\alpha\in[0,1]$为常量
3. 对初始值无偏的指数近因加权平均(**适用更广**) \
   使用修正步长
   $$
   \begin{align}
   &\beta_n=\alpha/\overline{o}_n \nonumber\\
   &\overline{o}_n=\overline{o}_{n-1}+\alpha(1-\overline{o}_{n-1}) \nonumber
   \end{align}
   $$
   此时既可应用于非稳态过程且无偏

### 乐观初始值
对动作的初始预期值可设较大，当采用贪心方法时，无论哪一种动作被选择，实际收益低于预期收益，均会使得学习器对收益失望，进而选择其他动作（因而进行大量试探）

### 梯度赌博机算法
**偏好函数**:$H_t(a)$越大，动作就越频繁被选择 \
对偏好函数使用$softmax$:
$$
Pr(A_t=a)=\frac{e^{H_t()a}}{\sum_{b=1}^k e^{H_t(b``)}}\triangleq\pi_t(a)
$$
$\pi_t(a)$可表示动作$a$在$t$时刻被选择概率
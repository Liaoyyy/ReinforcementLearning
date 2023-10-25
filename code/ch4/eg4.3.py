#赌徒问题
import numpy as np
from matplotlib import pyplot as plt
import time

state=list(range(0,101,1))#状态空间
p_h=0.25 #抛硬币正面朝上概率
stateValues=np.random.random(size=101) #状态价值函数
stateValues[0]=0
stateValues[100]=1
bound = 0.01

def initStrategy(step):
    strategy=[[0]]*(step-1)
    for i in range(1,step-1):
        strategy[i]=[np.random.randint(0,min(i,step-i-1))]
    return strategy

def generateRandomNum(p_h):
    num=np.random.random()
    if(num<=p_h):
        return 1#正面朝上
    else:
        return 0#背面朝上

#使用策略迭代方法
def DP(bound):
    #策略评估
    delta=1
    while(delta>bound):
        delta=0
        for s in state:
            v=stateValues[s]
            action=np.random.choice(strategy[s]) #根据策略函数做出选择
            stateValues[s]=p_h*(action+v)+(1-p_h)*(v)
            delta=max(delta,abs(v-stateValues[s]))

if __name__ == "__main__":
    strategy=initStrategy(101) #随机初始化策略函数
    startTime=time.time()
    DP(bound)
    endTime=time.time()
    print("Traning period="+(startTime-endTime))

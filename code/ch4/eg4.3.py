#赌徒问题
import numpy as np
from matplotlib import pyplot as plt
import time

target=10
state=list(range(0,target+1,1))#状态空间
p_h=0.25 #抛硬币正面朝上概率
stateValues=np.random.random(size=target+1) #状态价值函数
stateValues[0]=0
stateValues[target]=1
bound = 0.01
gamma = 0.9

def initStrategy(step):
    strategy=[[0]]*(step)
    for i in range(1,step-1):
        strategy[i]=[np.random.randint(0,min(i,step-i-1))]
    return strategy

def generateRandomNum(p_h):
    num=np.random.random()
    if(num<=p_h):
        return 1#正面朝上
    else:
        return 0#背面朝上

#单步策略改进
def searchBestAction(stateValues,curState):
    newStepStrat=[]#新策略
    allowedAction=list(range(0,min()))
    return newStepStrat

#使用策略迭代方法
def DP(bound,stateValues,strategy):

    while(1):
        #策略评估
        delta=1
        while(delta>bound):
            delta=0.0
            for s in state:
                v=stateValues[s]
                action=np.random.choice(strategy[s]) #根据策略函数做出选择
                stateValues[s]=p_h*(action+gamma*stateValues[s+action])+(1-p_h)*(-action+gamma*stateValues[s-action])
                delta=max(delta,abs(v-stateValues[s]))

        #策略改进
        policyStable=True
        for s in state:
            oldAction=strategy[s]



if __name__ == "__main__":
    strategy=initStrategy(target+1) #随机初始化策略函数
    startTime=time.time()
    DP(bound,stateValues,strategy)
    endTime=time.time()
    print("Traning period=",(endTime-startTime))

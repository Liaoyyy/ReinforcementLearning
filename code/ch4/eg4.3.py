#使用策略迭代方法解决赌徒问题

import numpy as np
from matplotlib import pyplot as plt
import time

target=10 #十步
state=list(range(0,target+1,1))#状态空间
p_h=0.4 #抛硬币正面朝上概率
stateValues=np.zeros(target+1) #状态价值函数
stateValues[0]=0
stateValues[target]=1
bound = 0.001
gamma = 0.9

def initStrategy(step):
    strategy=[[0]]*(step)
    for i in range(1,step-1):
        #strategy[i]=[np.random.randint(0,min(i,step-i-1))]
        strategy[i] = [0]
    return strategy

def generateRandomNum(p_h):
    num=np.random.random()
    if(num<=p_h):
        return 1#正面朝上
    else:
        return 0#背面朝上

#单步策略改进
def searchBestAction(stateValues,curState,strategy):
    newStepStrat=[]#当前状态下新策略
    temp=-100
    allowedAction=list(range(0,min(curState,target-curState)+1))
    for a in allowedAction:
        if(a+curState==target):
            reward=1
        else:
            reward=0
        Value=p_h*(gamma*stateValues[curState+a])+(1-p_h)*(gamma*stateValues[curState-a])
        if(Value>temp):
            newStepStrat=[a]
            temp=Value
        elif(Value==temp):
            if a not in newStepStrat:
                newStepStrat.append(a)
    return newStepStrat

#使用策略迭代方法
def DP(bound,stateValues,strategy):
    flag=0
    while(1):
        flag+=1
        #策略评估
        delta=1
        while(delta>bound):
            delta=0.0
            for s in state:
                if(s==target or s==0):
                    continue
                v=stateValues[s]
                action=np.random.choice(strategy[s]) #根据策略函数做出选择
                if(action+s==target):
                    reward=1
                else:
                    reward=0
                stateValues[s]=p_h*(gamma*stateValues[s+action])+(1-p_h)*(gamma*stateValues[s-action])
                delta=max(delta,abs(v-stateValues[s]))

        #策略改进
        policyStable=True
        for s in state:
            if (s == target or s == 0):
                continue
            oldAction=strategy[s].copy()
            strategy[s]=searchBestAction(stateValues,s,strategy)
            oldAction.sort()
            strategy[s].sort()
            if(strategy[s]==oldAction):
                continue
            else:
                policyStable = False

        if(policyStable):
            print(flag)
            return#已收敛至最优解，返回
if __name__ == "__main__":
    strategy=initStrategy(target+1) #随机初始化策略函数
    startTime=time.time()
    DP(bound,stateValues,strategy)#动态规划 策略改进
    endTime=time.time()
    print("Traning period=",(endTime-startTime))
    print(strategy)
    print(stateValues)
#例题4.2 杰克租车问题
import numpy as np
import pandas as pd

#Actions为杰克从A地挪至B地车辆多少,负数时为从B到A挪时车辆多少
Actions=list(range(-5,6))

stateValue=np.random.random(size=(20,20)) #初始化状态价值
strategy=np.random.randint(-5,6,size=(20,20)) #随机初始化策略函数
state=np.random.randint(0,20,size=(1,2))  #初始状态

#策略评估
delta = 1
while(delta > 0.01):
    for i in range(20):
        for j in range(20):
            value=stateValue[i][j]
            action=strategy[i][j]
            
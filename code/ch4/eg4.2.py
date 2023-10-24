#例题4.2 杰克租车问题
import numpy as np
import pandas as pd

#Actions为杰克从A地挪至B地车辆多少,负数时为从B到A挪时车辆多少
Actions=list(range(-5,6))

#初始化状态价值
stateValue=np.random.random(size=(20,20))


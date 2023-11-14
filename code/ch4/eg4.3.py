#Use policy iteration to solve gamblers problem

import numpy as np
import time
from matplotlib import pyplot as plt

# total 100 step
step = 100
# probability of facing up
p_h = 0.4
# decay
gamma = 1
# tolerance bound
bound = 0.000000001
epsilon = 0.0001

policy = np.zeros(step+1,dtype=int)
state = list(range(0,step+1))
stateValues = np.random.random(size=step+1)
stateValues[step] = 1

# policy evaluation
def Evaluation():
    while(1):
        delta = 0
        for s in state:
            if(s==0 or s==step):
                continue
            oldValue = stateValues[s]
            stateValues[s] = p_h * gamma * stateValues[(s+policy[s])] + (1-p_h) * gamma * stateValues[(s-policy[s])]
            delta = max(delta, abs(oldValue-stateValues[s]))
        if(delta < bound):
            break

# policy improvement
def Improvment():
    while(1):
        policy_stable = 1
        for s in state:
            if(s==0 or s==step):
                continue
            newPolicy = 0;
            maxValue = -1
            for action in range(1,min(step-s,s)+1):
                curValue = p_h * gamma * stateValues[s+action] + (1-p_h) * gamma * stateValues[s-action]
                if(curValue > maxValue+epsilon):
                    newPolicy = action
                    maxValue = curValue
            if(newPolicy != policy[s]):
                policy_stable = 0

            # update
            policy[s]=newPolicy

        if(policy_stable):
            return 0
        else:
            Evaluation()

def gamblers_problem():
    Improvment()

if __name__ == "__main__":
    gamblers_problem()
    # show the result
    print(policy)
    print(stateValues)
    plt.plot(state,policy)
    plt.show()
    plt.plot(state,stateValues)
    plt.show()
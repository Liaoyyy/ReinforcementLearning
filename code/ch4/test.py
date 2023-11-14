#Use policy iteration to solve gamblers problem

import numpy as np
import time

# total 100 step
step = 20
# probability of facing up
p_h = 0.4
# decay
gamma = 1
# tolerance bound
bound = 0.001

policy = np.zeros(step+1,dtype=int)
state = list(range(0,step+1))
stateValues = np.zeros(step+1)
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
        policy_stable = True
        for s in state:
            if(s==0 or s==step):
                continue
            newPolicy = 0;
            maxValue = -1
            for action in range(0,min(step-s,s)+1):
                curValue = p_h * gamma * stateValues[s+action] + (1-p_h) * gamma * stateValues[s-action]
                if(curValue > maxValue):
                    newPolicy = action
                    maxValue = curValue
            # update
            policy[s]=newPolicy
            stateValues[s]=maxValue
            if(newPolicy != policy[s]):
                policy_stable = False
        if(policy_stable):
            break
        else:
            Evaluation()

if __name__ == "__main__":
    Improvment()
    # show the result
    print(policy)
    print(stateValues)
import numpy as np
import pandas as pd
from IPython.display import clear_output
epsilon=0.1
def createQTable(stateNum,actionNum,actions):
    m=np.zeros([stateNum,actionNum],dtype=float)
    return pd.DataFrame(data=m,columns=actions)

def feedBack(state,action,maze):
    row,col = maze.shape
    state_new=[state[0],state[1]]
    if(action=='up'):
        state_new[0]-=1
    elif(action=='down'):
        state_new[0]+=1
    elif(action=='left'):
        state_new[1]-=1
    elif(action=='right'):
        state_new[1]+=1
    if(maze[state_new[0]][state_new[1]]==-1):
        R=1
    else:
        R=0
    return state_new,R

def printEnv(maze,state):
    row,col=maze.shape
    for i in range(row):
        print('\r| ',end='')
        for j in range(col):
            if(i==state[0] and j==state[1]):
                print('C ',end='')
            elif(maze[i][j]==-1):
                print('{} '.format('T'),end='')
            else:
                print('{} '.format(maze[i][j]),end='')
        print('|\n',end='')
    return

def chosseAction(Q_table,state,maze):
    _,col=maze.shape
    stateActions=Q_table.iloc[state[0]*col+state[1],:]
    if((np.random.uniform()<epsilon) | (stateActions==0).all()):
        action=np.random.choice(Q_table.columns.values)
        while not validAction(state,maze,action):
            action=np.random.choice(Q_table.columns.values)
    else:
        action=stateActions.idxmax()
    return action

def validAction(state,maze,action):
    row,col=maze.shape
    if (action == 'up' and (state[0] > 0) and (maze[state[0]-1][state[1]] != 1)):
        return True
    elif (action == 'down' and (state[0] + 1 < row) and (maze[state[0]+1][state[1]] != 1)):
        return True
    elif (action == 'left' and (state[1] > 0) and (maze[state[0]][state[1]-1] != 1)):
        return True
    elif (action == 'right' and (state[1] + 1 < col) and (maze[state[0]][state[1]+1] != 1)):
        return True
    else:
        return False
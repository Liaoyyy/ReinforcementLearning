0  # This is a sample Python script.
import pandas as pd
import numpy as np
import Q_learning as ql
import time

# Create a maze num '1' stands for a block and num'-1' refers to the destination
maze = np.array([[0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [1, 0, 1, -1]])

Actions = ['up', 'down', 'left', 'right']
epoch = 80
row, col = maze.shape
gamma = 0.8  # discount factor
alpha = 0.2  # learning rate
shortestPath=[];
shortestLen=1000;
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Q_table = ql.createQTable(maze.size, len(Actions), Actions)
    for i in range(epoch):
        state = [0, 0]
        stepCounter = 0
        is_terminated = False

        path=[state]
        while not is_terminated:
            action = ql.chosseAction(Q_table, state, maze)
            newState, R = ql.feedBack(state, action, maze)
            q_current = Q_table.loc[(state[0] * col + state[1]), action]
            if (maze[newState[0]][newState[1]] == -1):
                # Reach the destination
                TD_target = R
                is_terminated = True
            else:
                # hasn't reached the destination
                TD_target = R + gamma * Q_table.iloc[newState[0] * col + newState[1], :].max()
            TD_error = TD_target - q_current
            Q_table.loc[(state[0] * col + state[1]), action] += alpha * TD_error
            state = newState.copy()
            path.append(state)
            stepCounter += 1
        if(stepCounter<shortestLen):
            shortestPath=path.copy()
            shortestLen=stepCounter
        print(stepCounter)
    print(Q_table,end='\n\n')
    print(maze)
    print('\n'+'Shortest path: ',end='')
    print(path)

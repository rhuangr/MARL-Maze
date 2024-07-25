# from maze_agent import Agent
# from maze import Maze
# from collections import deque

# def main():
#     maze = Maze(rand_range=[10,14], rand_start=False, hardcore=True)
#     maze.agents.brain.train()
# main()
import numpy as np
import time
cur_time = time.time()
for i in range(100000):
    x = np.zeros(4)
    y = [0,0]
    y.extend(x)
after = time.time()
print(after-cur_time)
print()
cur_time = time.time()
for i in range(100000):
    x = [0,0,0,0]
    y = [0,0]
    y.extend(x)
after = time.time()
print(after-cur_time)

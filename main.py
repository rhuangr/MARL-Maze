from agent import Agent
from maze import Maze
from collections import deque

def main():
    maze = Maze(rand_range=[15,15], rand_start=False, hardcore=True)
    maze.agent.brain.train()
main()


# x = deque([3],maxlen=2)
# x.append(2)
# print(x)
# x.append(2)
# print(x)
# x.append(3)
# print(x)
# x.append(4)
# print(x)
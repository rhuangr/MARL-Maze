from agent import Agent
from maze import Maze
from collections import deque

def main():
    maze = Maze(rand_range=[10,14], rand_start=False, hardcore=True)
    maze.agent.brain.train()
main()
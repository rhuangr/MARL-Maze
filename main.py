from agent import Agent
from maze import Maze
import time

def main():
    maze = Maze()
    maze.agent.brain.train(maze_size_range=[4,15])
# main()

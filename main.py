from agent import Agent
from maze import Maze

def main():
    maze = Maze(rand_range=[3,5])
    maze.agent.brain.train()
main()
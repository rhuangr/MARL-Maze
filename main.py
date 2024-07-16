from agent import Agent
from maze import Maze

def main():
    maze = Maze(rand_range=[8,12], rand_start=True, hardcore=True)
    maze.agent.brain.train()
main()
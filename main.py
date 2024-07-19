from agent import Agent
from maze import Maze

def main():
    maze = Maze(rand_range=[4,5], rand_start=False, hardcore=False)
    maze.agent.brain.train()
main()
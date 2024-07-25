from maze import Maze
from maze_agent import Agent
from pygame import Color
from PPO import PPO

RED =  Color("red")
PALE_RED = Color("palevioletred1")

BLUE = Color("royalblue1")
PALE_BLUE = Color("darkslategray1")

YELLOW = Color("gold1")
PALE_YELLOW = Color("khaki1")


brain = PPO()
agents = (Agent('RED', brain, RED, PALE_RED, 2),
            Agent('YELLOW', brain, YELLOW, PALE_YELLOW, 3),
            Agent('BLUE',brain, BLUE, PALE_BLUE, 4))
maze = Maze(agents=agents, rand_range=[2,2], rand_start=False, hardcore=False) 
brain.train()
# maze.display_policy()
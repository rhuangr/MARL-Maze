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


brain = PPO(agent_amount=2)
agents = (Agent('RED', brain, RED, PALE_RED, 2),
        Agent('BLUE',brain, BLUE, PALE_BLUE, 3))
maze = Maze(agents=agents, max_timestep=2500, rand_sizes=True, rand_range=[5,8], rand_start=False, hardcore=True, default_size=[4,4]) 
brain.train()
# maze.display_policy()

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

# although most code is dynamic to the amount of agents, not all of it is.
# please do not change the amount of agents in the maze unless you want a disaster...
brain = PPO(agent_amount=2, batch_size=15000, lr=0.00014)
agents = (Agent('RED', brain, RED, PALE_RED, 2),
        Agent('BLUE',brain, BLUE, PALE_BLUE, 3))
maze = Maze(agents=agents, max_timestep=1200, rand_sizes=True, rand_range=[12,13], rand_start=True, difficulty=1, default_size=[4,4]) 

# brain.train()
maze.display_policy() 

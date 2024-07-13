import time
import random
from PPO import PPO

class Agent:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        
        self.brain = 1
        self.direction = 2 # value at index of ['north', 'east, 'south', 'west']
        self.tag = 2

    def get_action(self):
        pass
        return random.randint(0,7)


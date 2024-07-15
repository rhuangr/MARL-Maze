import PPO
import math

class Agent:
    def __init__(self, x, y, color, maze):
        self.x = x
        self.y = y
        self.color = color      
        self.brain = PPO.PPO(maze=maze)
        self.direction = 2 # value at index of ['north', 'east, 'south', 'west']
        self.tag = 2

    def get_action(self, obs, mask):
        action, prob , _ = self.brain.get_action(obs, mask)
        return action, math.exp(prob)


import PPO
import math

class Agent:
    def __init__(self, x, y, color, maze):
        self.x = x
        self.y = y
        self.color = color      
        self.brain = PPO.PPO(maze=maze)
        self.direction = 2 # direction facing value at index of ['north', 'east, 'south', 'west']
        self.tag = 2

        self.min_x_visited = x
        self.max_x_visited = x
        self.min_y_visited = y
        self.max_y_visited = y 
        
        # agent's estimate of dimensions W x H of the maze
        self.width_estimate = 1
        self.height_estimate = 1
        
    def get_action(self, obs, mask):
        action, prob , _ = self.brain.get_action(obs, mask)
        return action, math.exp(prob)
    
    def estimate_maze(self, direction):
        
        # print(f"current: {self.x}, {self.y}")
        # print()
        # print(f"x: {self.max_x_visited}, {self.min_x_visited}")
        # print(f"y: {self.max_y_visited}, {self.min_y_visited}")
        # print(f"direcotion: {direction}, {self.direction}")
        updated = False
        if direction == 0 and self.y < self.min_y_visited :
            self.min_y_visited = self.y
            updated = True
        elif direction == 1 and self.x > self.max_x_visited:
            self.max_x_visited = self.x
            updated = True
        elif direction == 2 and self.y > self.max_y_visited:
            self.max_y_visited = self.y
            updated = True
        elif direction == 3 and self.x < self.min_x_visited:
            self.min_x_visited = self.x
            updated = True

        # print(f"1 + {self.max_x_visited} - {self.min_x_visited}")
        # print(f"1 + {self.max_y_visited}- {self.min_y_visited}")
        # ommited + 1 from the estimate calculations to avoid redundance in relative position calculations
        self.width_estimate = self.max_x_visited - self.min_x_visited
        self.height_estimate = self.max_y_visited - self.min_y_visited
        
        return updated
        
        # print(f"w estim: {self.width_estimate}, h estim: {self.height_estimate}")

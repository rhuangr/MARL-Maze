import PPO
from math import exp
from collections import deque

AGENT_VISION_RANGE = 4
ACTIONS = ['forward', 'right', 'backward', 'left']
DIRECTIONS = ['north', 'east', 'south', 'west']
BINARY_DIRECTIONS = [[0,0], [0,1], [1,0], [1,1]] # binary representation of possible directions
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # change in x,y after moving in respective cardinal direction


FEATURE_NAMES = ['direction', 'dead ends', 'visible unmarked cell', 'visible_end',
                 'move t-4', 'move t-3','move t-2','move t-1','timestep', 'relative x', 'relative y']
FEATURE_DIMS = [4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1]

class Agent:
    def __init__(self, color, maze, breaks=2):
        self.x = 0
        self.y = 0
        self.maze = maze
        self.color = color      
        self.brain = PPO.PPO(self, maze)
        self.direction = 2 # direction facing value at index of ['north', 'east, 'south', 'west']
        self.tag = 2

        self.total_steps = 0
        self.memory = deque([10,10,10,10], maxlen=4)
        self.average_exit = 5000
                
        # agent's estimate of dimensions W x H of the maze
        self.width_estimate = 1
        self.height_estimate = 1
        self.reset_estimates()

    def reset(self):
        self.x, self.y = self.maze.start
        self.reset_estimates()
        self.direction = 2
        self.memory = deque([10,10,10,10], maxlen=4)
        # self.breaks_remaining = self.max_breaks
        
    def get_action(self, obs, mask):
        action, prob , _ = self.brain.get_action(obs, mask)
        return action, exp(prob)
    
    def move(self, x, y, direction):
        self.x, self.y = x, y
        self.direction = direction
        return self.estimate_maze(direction)
    
    def get_observations(self):
        
        # start building the observation vector
        direction = [0,0,0,0]
        direction[self.direction] = 1
        dead_ends, move_action_mask, walls = self.get_dead_ends()
        # print(dead_ends)
        visible_marked, visible_end = self.get_visibility_features()
        memory = self.get_memory()

        features = [direction, dead_ends, visible_marked, visible_end, memory]
        observations = []
        for feature in features:
            observations.extend(feature)
            
        timestep = 1/self.average_exit * self.total_steps    
        observations.append(timestep)
        
            # since this project relies on agents not knowing the layout of the maze
            # rel x, rel y represent the agent's estimate of his current position x,y
        relative_x = 0 if self.width_estimate < 3 else (self.x - self.min_x_visited) / self.width_estimate
        relative_y = 0 if self.height_estimate < 3 else (self.max_y_visited - self.y) / self.height_estimate
        observations.append(relative_x)
        observations.append(relative_y)

        # start building the action mask
        action_mask = []
        mark_action_mask = True if self.maze.layout[self.y][self.x] != self.tag else False
        break_action_mask = [False, False, False, False] #if self.breaks_remaining == 0 else walls
        break_action_mask[2] = mark_action_mask
        action_mask.extend(move_action_mask)
        action_mask.extend(break_action_mask)
        
        # print(f"dead ends: {dead_ends}, mask: {action_mask}")
        # print(f"marks: {visible_marked}, unmark: {visible_unmarked}")
        # print(f"end vis: {visible_end}")
        # print(f"rel x: {relative_x}, rel y: {relative_y}")
        # print()
        return observations, action_mask
    
    # returns a list representing visible dead ends in all four directions
    def get_dead_ends(self):

        # binary representation of: is there a dead end in any clockwise cardinal directions, starting from north
        dead_ends = [0,0,0,0] 
        current_cell = (self.x, self.y)
        neighbors, walls = self.get_neighbors(current_cell)
        move_action_mask = neighbors

        # if there is a wall in the adjacent cell of a given direction, then that direction is a dead end
        for direction in range(len(neighbors)):
            if neighbors[direction] == False:
                dead_ends[direction] = 1

        # chooses a direction to expand vision
        for direction in range(len(DELTAS)): 
            
            # if there is a dead end in direction i, then continue to next direction
            if dead_ends[direction] == 1:
                    continue
                
            next_x, next_y = self.x, self.y
            x_dif, y_dif = DELTAS[(direction+self.direction)%4][0], DELTAS[(direction+self.direction)%4][1]

            # simulates vision, by checking if next cells in the current direction are dead ends
            for j in range(AGENT_VISION_RANGE):
                next_x, next_y = next_x + x_dif, next_y + y_dif
                neighbors, _ = self.get_neighbors((next_x, next_y))
                
                # if the cell we expand in has a turn, it is not a dead end
                if neighbors[(direction + 1)%4] == True or neighbors[(direction - 1)%4] == True:
                    break

                # if a cell is blocked in 3 directions, then that cell is a dead end
                if neighbors.count(True) == 1:
                    dead_ends[direction] = 1
                    break

                # if the direction in which the search is expanding in encounters a wall, then no need to keep expanding
                elif neighbors[direction] == False:
                    break

        return dead_ends, move_action_mask, walls
        
    def get_visibility_features(self):
        mark = self.tag
        visible_marked_squares = [0,0,0,0]
        visible_end = [0,0,0,0]
        
        for i in range(len(DELTAS)):
            next_x, next_y = self.x, self.y
            x_dif, y_dif = DELTAS[(i+self.direction)%4][0], DELTAS[(i+self.direction)%4][1]
            for j in range(AGENT_VISION_RANGE):
                next_x, next_y = next_x + x_dif, next_y + y_dif
                if next_x < 0 or next_x >= self.maze.width or next_y < 0 or next_y >= self.maze.height:
                    break

                new_cell = self.maze.layout[next_y][next_x]

                if (next_x, next_y) == self.maze.end:
                    visible_end[i] = 1

                if new_cell == mark:
                    visible_marked_squares[i] = 1

                if new_cell == 1:
                    break

        return visible_marked_squares, visible_end
    
    def get_memory(self):
        memory_feature = [0 for _ in range(16)]
        for i in range(4):
            action = self.memory[i]
            if action < 4:
                memory_feature[i*4+action] = 1
        return memory_feature
    
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
    
    def reset_estimates(self):
        self.min_x_visited = self.x
        self.max_x_visited = self.x
        self.min_y_visited = self.y
        self.max_y_visited = self.y

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = [False, False, False, False]
        walls = [False, False, False, False]

        for i in range(len(DELTAS)):
            x_dif, y_dif = DELTAS[(i + self.direction)%4]
            neighbor_x, neighbor_y = x + x_dif, y + y_dif

            if (0 <= neighbor_x < self.maze.width and 0 <= neighbor_y < self.maze.height):
                if self.maze.layout[neighbor_y][neighbor_x] == 1:
                    walls[i] = True
                else:
                    neighbors[i] = True

        return neighbors, walls
    
    def print_obs(self):
        obs, _ = self.get_observations()
        index = 0
        for i in range(len(FEATURE_DIMS)):
            print(f"{FEATURE_NAMES[i]}: {obs[index:index+FEATURE_DIMS[i]]}")
            index+=FEATURE_DIMS[i]
        print()
        
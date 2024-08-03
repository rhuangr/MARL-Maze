from PPO import PPO
import numpy as np
from math import exp
from collections import deque

AGENT_VISION_RANGE = 4
ACTIONS = ['forward', 'right', 'backward', 'left']
DIRECTIONS = ['north', 'east', 'south', 'west']
BINARY_DIRECTIONS = [[0,0], [0,1], [1,0], [1,1]] # binary representation of possible directions
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # change in x,y after moving in respective cardinal direction


FEATURE_NAMES = ['Direction', 'Dead Ends', 'Own Mark Visible', 'Others Mark Visible', 'Agent Visible', 'Others Direction','Visible Key',
                 'Move t-4', 'Move t-3', 'Move t-2', 'Move t-1', 'Relative Position', 'Other Agent Relative Position',
                 'Sees End', 'End Direction','Visible Agent Knows End','Has Key', 'Visible Agent Has key','Timestep', 'ID']
FEATURE_DIMS = [4,4,4,4,4,4,4,4,4,4,4,2,2,1,4,1,1,1,1,2]

class Agent:
    def __init__(self, name, brain, color, mark_color, tag):
        self.maze = None
        self.name = name
        self.brain = brain
        self.color = color
        self.mark_color = mark_color

        self.x = 0
        self.y = 0
        self.direction = 2 # direction facing value at index of ['north', 'east, 'south', 'west']
        self.tag = tag
        self.knows_end = False
        self.sees_end = False
        self.has_key = False
        self.sees_key = False
        self.other_last_seen = None
        self.visited_cells = {}
        
        # signal fields
        # self.signal_time = 0 # determines the radius of signal shape drawn on screen
        # self.is_signalling = False
        # self.signal_origin = None

        self.current_t = 0
        self.memory = deque([-1,-1,-1,-1], maxlen=4)
        self.average_exit = 5000
                
        # agent's estimate of dimensions W x H of the maze
        self.min_x_visited = self.x
        self.max_x_visited = self.x
        self.min_y_visited = self.y
        self.max_y_visited = self.y
        self.width_estimate = 1
        self.height_estimate = 1

    def reset(self,x,y):
        self.current_t = 0
        self.x, self.y = x,y
        self.other_last_seen = (x,y)
        self.reset_estimates()
        self.direction = 2
        self.visited_cells = {}
        self.memory = deque([-1,-1,-1,-1], maxlen=4)
        self.signal_time = 0
        self.is_signalling = False
        self.signal_origin = None
        self.knows_end = False
        self.sees_end = False
        self.end_direction = None
        self.has_key = False
        self.sees_key = False
        
    def get_action(self, obs, mask):
        action, prob= self.brain.get_action(obs, mask)
        # print(f'name: {self.name}')
        return action, exp(prob)
    
    def move(self, x, y, direction):
        self.x, self.y = x, y
        self.direction = direction
        
    def get_observations(self):
        # start building the observation vector
        direction = [0,0,0,0]
        direction[self.direction] = 1
        (visible_own_mark, visible_others_mark, visible_agents, visible_key,
         other_has_key, other_knows_end, other_dir, other_last_pos) = self.get_visibility_features()
        dead_ends, move_action_mask = self.get_dead_ends()
        memory = self.get_memory()

        features = [direction, dead_ends, visible_own_mark, visible_others_mark,
                    visible_agents, other_dir,visible_key, memory]
        observations = []
        for feature in features:
            observations.extend(feature)
            # since this project relies on agents not knowing the layout of the maze
            # rel x, rel y represent the agent's estimate of his current position x,y
        relative_x = (self.x - self.min_x_visited) / self.width_estimate
        relative_y = (self.max_y_visited - self.y) / self.height_estimate
        observations.extend([relative_x, relative_y])
        observations.extend(other_last_pos)

        # SIGNAL RELATED FEATURE
        # signal_direction = []
        # for agent in self.maze.agents:
        #     if agent == self:
        #         continue
        #     if agent.is_signalling:
        #         signal_direction.extend(self.get_direction(agent.signal_origin))
        #     else:
        #         signal_direction.extend([0,0,0,0])
        # observations.extend(signal_direction)

        observations.append(self.sees_end)
        end_direction = [0,0,0,0] if self.knows_end == False else self.get_direction_from(self.maze.end)
        self.end_direction = end_direction
        observations.extend(end_direction)
        observations.append(other_knows_end)
        observations.append(self.has_key)
        observations.append(other_has_key)
        observations.append(self.current_t/self.maze.max_timestep)
        id = [0,0]
        id[2-self.tag] = 1
        observations.extend(id)

        # start building the action mask
        mark_action_mask = True if self.maze.layout[self.y][self.x] != self.tag else False
        action_mask = move_action_mask
        action_mask.append(mark_action_mask)
        # action_mask.append(signal_action_mask)
            
        return observations, action_mask
    
    # returns a list representing visible dead ends in all four directions
    def get_dead_ends(self):
        # binary representation of: is there a dead end in any clockwise cardinal directions, starting from north
        dead_ends = [0,0,0,0]
        neighbors = self.get_neighbors((self.x, self.y))
        move_action_mask = neighbors
        distance = 1/AGENT_VISION_RANGE
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
            for j in range(1,AGENT_VISION_RANGE+1):
                next_x += x_dif
                next_y += y_dif
                neighbors = self.get_neighbors((next_x, next_y))
                
                # if the cell we expand in has a turn, it is not a dead end
                if neighbors[(direction + 1)%4] == True or neighbors[(direction - 1)%4] == True:
                    break

                # if a cell is blocked in 3 directions, then that cell is a dead end
                if neighbors.count(True) == 1:
                    dead_ends[direction] = 1 - j*distance
                    break

                # if the direction in which the search is expanding in encounters a wall, then no need to keep expanding
                elif neighbors[direction] == False:
                    break 
        if self.sees_end != True and self.sees_key != True:
             move_action_mask = [dead_end==0 for dead_end in dead_ends]
        # print(f'name: {self.name} mask: {move_action_mask}, sees end{self.sees_end}')
        return dead_ends, move_action_mask
        
    def get_visibility_features(self):
        visible_own_mark = [0,0,0,0]
        visible_others_mark = [0,0,0,0]
        visible_agents = []
        visible_end = [0,0,0,0]
        visible_key = [0,0,0,0]
        visible_agent_direction = [0,0,0,0]
        visible_agent_has_key = 0.5
        visible_agent_knows_end = 0.5
        self.sees_end = False
        self.sees_key = False
        
        for agent in self.maze.agents:
            if agent == self:
                continue
            if (agent.x,agent.y) == (self.x,self.y):
                visible_agents.extend([1,1,1,1])
                self.other_last_seen = (agent.x, agent.y)
                visible_agent_has_key = agent.has_key
                visible_agent_knows_end = agent.knows_end
                visible_agent_direction[agent.direction] = 1
        
        for dir in range(len(DELTAS)):
            next_x, next_y = self.x, self.y
            x_dif, y_dif = DELTAS[(dir+self.direction)%4][0], DELTAS[(dir+self.direction)%4][1]
            for j in range(1,AGENT_VISION_RANGE+1):
                # increment x,y to expand vision in that direction
                next_x += x_dif
                next_y += y_dif
                # if new x,y is out of bounds or hits a wall, break
                if (next_x < 0 or next_x >= self.maze.width or next_y < 0 or next_y >= self.maze.height or
                     self.maze.layout[next_y][next_x] == 1):
                    break
                # check for end
                if (next_x, next_y) == self.maze.end:
                    self.knows_end = True
                    self.sees_end = True
                    visible_end[dir] = 1
                # check for key
                if (next_x, next_y) == self.maze.key:
                    self.sees_key = True
                    visible_key[dir] = 1
                # check for visible agents and its features
                agents_in_position = self.maze.agent_positions.get((next_x, next_y))
                if agents_in_position != None:
                    for agent in agents_in_position:
                        if agent == self:
                            continue
                        self.other_last_seen = (agent.x, agent.y)
                        visible_agent_has_key = agent.has_key
                        visible_agent_direction[agent.direction] = 1
                        visible_agent_knows_end = agent.knows_end
                        agent_direction = [0,0,0,0]
                        agent_direction[dir] = 1
                        visible_agents.extend(agent_direction)
                # check for own marks
                if self.maze.layout[next_y][next_x] == self.tag:
                    visible_own_mark[dir] = 1
                # check for other agent's marks
                elif self.maze.layout[next_y][next_x] > 1:
                    visible_others_mark[dir] = 1
                # update min max visited
                self.update_maze_minmax((dir+self.direction)%4, next_x, next_y)
        
        self.update_maze_dims()
        other_relative_x = (self.other_last_seen[0] - self.min_x_visited) / self.width_estimate
        other_relative_y = (self.max_y_visited - self.other_last_seen[1]) / self.height_estimate
        agent_last_pos = [other_relative_x, other_relative_y]
        visible_agents.extend([0 for _ in range((len(self.maze.agents) - 1) * 4-len(visible_agents))])
        return( visible_own_mark, visible_others_mark, visible_agents, visible_key, visible_agent_has_key,
               visible_agent_knows_end, visible_agent_direction, agent_last_pos)
    
    def update_visited_cells(self):
        pos = (self.x,self.y)
        if pos in self.visited_cells:
            self.visited_cells[pos] += 1 
            return self.visited_cells[pos]
        else:
            self.visited_cells[pos] = 1
            return 1
        
    def get_memory(self):
        memory_feature = [0 for _ in range(16)]   
        for i, move in enumerate(self.memory):
            if move > -1:
                memory_feature[i * 4 + move] = 1
        return memory_feature
    
    # gets the direction from the given point
    def get_direction_from(self, origin):
        if origin == (self.x, self.y):
            return [1,1,1,1]
        
        direction = [0,0,0,0]
        if origin[1] > self.y:
            direction[(2-self.direction)%4] = 1
        elif origin[1] < self.y:
            direction[(0-self.direction)%4] = 1
        if origin[0] > self.x:
            direction[(1-self.direction)%4] = 1
        elif origin[0] < self.x:
            direction[(3-self.direction)%4] = 1

        return direction
    
    def update_maze_minmax(self, direction, new_x, new_y):

        updated = False
        if direction == 0 and new_y < self.min_y_visited :
            self.min_y_visited = new_y
            updated = True
        elif direction == 1 and new_x > self.max_x_visited:
            self.max_x_visited = new_x
            updated = True
        elif direction == 2 and new_y > self.max_y_visited:
            self.max_y_visited = new_y
            updated = True
        elif direction == 3 and new_x < self.min_x_visited:
            self.min_x_visited = new_x
            updated = True
        # print(f"name: {self.name}, minx: {self.min_x_visited}, maxx: {self.max_x_visited}, miny: {self.min_y_visited}, maxy: {self.max_y_visited}")
        return updated
    
    def update_maze_dims(self):
                # ommited + 1 from the estimate calculations to avoid redundance in relative position calculations
        new_width_estimate = self.max_x_visited - self.min_x_visited 
        new_height_estimate = self.max_y_visited - self.min_y_visited
        self.width_estimate = new_width_estimate if new_width_estimate != 0 else 1
        self.height_estimate = new_height_estimate if new_height_estimate != 0 else 1
        # print(f"AGENT: {self.name} WIDTH,HEIGHT ESTIM: {self.width_estimate}, {self.height_estimate}")
        
    def reset_estimates(self):
        self.min_x_visited = self.x
        self.max_x_visited = self.x
        self.min_y_visited = self.y
        self.max_y_visited = self.y
        self.width_estimate = 1
        self.height_estimate = 1
   
    def get_neighbors(self, cell):
        x, y = cell
        neighbors = [False, False, False, False]

        for i in range(len(DELTAS)):
            x_dif, y_dif = DELTAS[(i + self.direction)%4]
            neighbor_x, neighbor_y = x + x_dif, y + y_dif

            if (self.maze.is_valid_cell(neighbor_x,neighbor_y) and self.maze.layout[neighbor_y][neighbor_x] != 1):
                neighbors[i] = True

        return neighbors
    
    def print_obs(self, obs):
        index = 0
        print(f"-------------- Agent {self.name} --------------")
        print(f"Current Position: {self.x}, {self.y}")
        for i in range(len(FEATURE_DIMS)):
            print(f"{FEATURE_NAMES[i]}: {obs[index:index+FEATURE_DIMS[i]]}")
            index+=FEATURE_DIMS[i]
        print(f"signal origin: {self.signal_origin}")
        print("----------------------------------------")
        print()
        
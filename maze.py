import pygame
import random
import time
from agent import Agent


# Initialize Pygame
pygame.init()

# Colors
WHITE = pygame.Color("white") # EMPTY CELL COLOR
BLACK =  pygame.Color("black") # WALL COLOR
YELLOW = pygame.Color("gold1") # START, END, SHORTEST PATH COLOR
RED =  pygame.Color("red")
PALE_RED = (255, 100, 150)
GREEN = (0, 255, 0)

CELL_SIZE = 20
AGENT_RADIUS = CELL_SIZE/2.2
AGENT_EYE_RADIUS = AGENT_RADIUS//4

TIMESTEP_LENGTH = 0.07 # USED WHEN RENDERING THE GAME
AGENT_VISION_RANGE = 4

ACTIONS = ['forward', 'right', 'backward', 'left']
DIRECTIONS = ['north', 'east', 'south', 'west']
BINARY_DIRECTIONS = [[0,0], [0,1], [1,0], [1,1]] # binary representation of possible directions

# TUPLES REPRESENTING THE CHANGE IN A CELL'S X,Y AFTER MOVING IN A CARDINAL DIRECTION
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class Maze:
    def __init__(self, max_timestep = 5_000, hardcore=False, rand_start=True, rand_sizes=True, rand_range=[6,12], default_size = [8,8]):

        # maze characteristics
        self.width = default_size[0] * 2 - 1
        self.height = default_size[1] * 2 - 1
        self.maze = None
        self.start = (0,0)
        self.end = None
        self.shortest_path = []
        self.shortest_path_len = 0

        self.observation_space = 20
        self.action_space = 8
        self.max_timestep = max_timestep

        self.agent = Agent(self.start[0], self.start[1], RED, self)
        # self.agent = Agent(16, 10, RED, self)

        self.current_t = 0
        self.last_timestep = time.time()

        # maze generation parameters
        self.rand_gen = rand_sizes
        self.rand_range = rand_range
        self.rand_start = rand_start
        self.hardcore=hardcore
        self.default_size = default_size
        

    def reset(self):
        self.current_t = 0
        self.agent.direction = 0
        
        if self.rand_gen == True:
            self.height = random.randint(self.rand_range[0], self.rand_range[1]) * 2 - 1
            self.width = random.randint(self.rand_range[0], self.rand_range[1]) * 2 - 1

        self.maze = [[1 for i in range(self.width)] for j in range(self.height)]
        self.build_maze()
        self.agent.x, self.agent.y = self.start
        return self.get_observations()

    def build_maze(self):
        
        # if random start mode, the start is chosen randomly instead of default 0,0
        if self.rand_start == True:
            start_x = random.randint(0, (self.width - 1)//2) * 2
            start_y = random.randint(0, (self.height - 1)//2) * 2
            self.start = (start_x, start_y)
        
        # building logic starts here
        stack = [self.start]
        while stack:
            current_cell = stack[-1]
            self.maze[current_cell[1]][current_cell[0]] = 0
            neighbors = self.get_wall_neighbors(current_cell)

            # if there are valid neighbors, we expand the maze in the direction of the randomly chosen neighbor
            if neighbors:
                next_cell = random.choice(neighbors)
                self.remove_wall(current_cell, next_cell)
                stack.append(next_cell)

            # no valid neighbors, therefore we backtrack until a cell with valid neighbors to continue expanding
            else:
                stack.pop()

        # if not hardcore mode, choose a random end
        if self.hardcore == False:
            self.set_end()
            self.shortest_path = self.get_shortest_path()
            self.shortest_path_len = len(self.shortest_path)
            return
        
        # if hardcore mode enabled, set end 5 times and choose the end which yields the longest shortest path
        max_length = 0
        lengths_ends = {}
        lengths_paths = {}
        for _ in range(6):
            self.set_end()
            shortest_path = self.get_shortest_path()
            shortest_path_len = len(shortest_path)
            max_length = max(max_length, shortest_path_len)
            lengths_ends[shortest_path_len] = self.end
            lengths_paths[shortest_path_len] = shortest_path

        self.end = lengths_ends[max_length]
        self.shortest_path = lengths_paths[max_length]
        self.shortest_path_len = max_length

        # count = 0 
        # for lists in self.maze:
        #     count += lists.count(0)
        # print(f"number of paths: {count} and shortest path length: {self.shortest_path_len}")

    def get_wall_neighbors(self, cell):
        x, y = cell
        neighbors = []

        for x_dif, y_dif in DELTAS:
            neighbor_x, neighbor_y = x + x_dif*2, y + y_dif*2

            if (0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height and
                self.maze[neighbor_y][neighbor_x] == 1):
                neighbors.append((neighbor_x, neighbor_y))

        return neighbors

    def get_path_neighbors(self, cell):
        x, y = cell
        neighbors = []

        for i in range(len(DELTAS)):
            x_dif, y_dif = DELTAS[(i + self.agent.direction)%4]
            neighbor_x, neighbor_y = x + x_dif, y + y_dif

            if (0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height and
                self.maze[neighbor_y][neighbor_x] != 1):
                neighbors.append(True)
            
            else:
                neighbors.append(False)

        return neighbors

    def remove_wall(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        self.maze[(y1 + y2) // 2][(x1 + x2) // 2] = 0

    def set_end(self):

        while True:
            end_x = random.randint(0, self.width - 1)
            end_y = random.randint(0, self.height - 1)
            temp_end = [end_x, end_y]
            coin = random.randint(0,1)
            end_location = self.width - 1 if coin == 0 else self.height - 1
            temp_end[coin] = end_location if random.randint(0,1) == 0 else 0
            print(f"{temp_end} and {self.width},{self.height}")
            print()
            if self.maze[temp_end[1]][temp_end[0]] == 0:
                self.end = (temp_end[0], temp_end[1])
                break

    def draw_maze(self):

        self.screen.fill(WHITE)
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                if self.maze[y][x] == 2:
                    pygame.draw.rect(self.screen, PALE_RED, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for x,y in self.shortest_path:
            center = ( x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(self.screen, YELLOW, center, CELL_SIZE//4)
        
        # Draw start
        pygame.draw.rect(self.screen, YELLOW, (self.start[0]*CELL_SIZE, self.start[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        # Draw end
        pygame.draw.rect(self.screen, YELLOW, (self.end[0]*CELL_SIZE, self.end[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        self.draw_agent()
        pygame.display.flip()

    def print_maze(self):
        for i in range(len(self.maze)):
            print(self.maze[i])

    def draw_agent(self):

        x = self.agent.x * CELL_SIZE + CELL_SIZE//2
        y = self.agent.y * CELL_SIZE + CELL_SIZE//2
        agent_center = (x,y)

        if self.agent.direction == 0:
            eye_center1 = (x + CELL_SIZE//5, y - CELL_SIZE//5)
            eye_center2 = (x - CELL_SIZE//5, y - CELL_SIZE//5)
        elif self.agent.direction == 1:
            eye_center1 = (x + CELL_SIZE//5, y + CELL_SIZE//5)
            eye_center2 = (x + CELL_SIZE//5, y - CELL_SIZE//5)
        elif self.agent.direction == 2:
            eye_center1 = (x + CELL_SIZE//5, y + CELL_SIZE//5)
            eye_center2 = (x - CELL_SIZE//5, y + CELL_SIZE//5)
        elif self.agent.direction == 3:
            eye_center1 = (x - CELL_SIZE//5, y + CELL_SIZE//5)
            eye_center2 = (x - CELL_SIZE//5, y - CELL_SIZE//5)

        pygame.draw.circle(self.screen, self.agent.color, agent_center, AGENT_RADIUS)
        pygame.draw.circle(self.screen, BLACK, eye_center1, AGENT_EYE_RADIUS)
        pygame.draw.circle(self.screen, BLACK, eye_center2, AGENT_EYE_RADIUS)

    def set_screen(self):
        screen_width = self.width * CELL_SIZE
        screen_height = self.height * CELL_SIZE
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Random Maze Generator")

    def step(self, action):
        self.current_t += 1

        # action logic
        if action > 3:
            self.maze[self.agent.y][self.agent.x] = self.agent.tag

        adjusted_action = (action + self.agent.direction) % 4
        x_dif, y_dif = DELTAS[adjusted_action]
        new_x, new_y = self.agent.x + x_dif, self.agent.y + y_dif

        if (0 <= new_x < self.width and 0 <= new_y < self.height and self.maze[new_y][new_x] != 1):
            self.agent.x = new_x
            self.agent.y = new_y
            # print(f"facing: {DIRECTIONS[self.agent.direction]}, action taken: {action} and {ACTIONS[adjusted_action]}, new x,y: {self.agent.x},{self.agent.y}")

        self.agent.direction = adjusted_action

        # reward function
        if (self.agent.x, self.agent.y) == self.end:
            reward = 1
        elif self.maze[self.agent.y][self.agent.x] == 0:
            reward = 0.009
        else:
            reward = 0

        # done/ truncated logic
        if (self.agent.x, self.agent.y) == self.end or self.current_t >= self.max_timestep:
            done = True
        else:
            done = False

        observations, action_mask = self.get_observations()
        return observations, action_mask, reward, done
    
    def get_observations(self):
        
        direction = self.get_agent_direction()
        dead_ends, action_mask = self.get_dead_ends()
        
        mark_action_mask = [True] * 4 if self.maze[self.agent.y][self.agent.x] != self.agent.tag else [False] * 4
        action_mask.extend(mark_action_mask)
        
        visible_marked, visible_unmarked, visible_end = self.get_visibility_features()
        on_marked_square = 1 if self.maze[self.agent.y][self.agent.x] == self.agent.tag else 0
        timestep = 0.0005 * self.current_t

        features = [direction, dead_ends, visible_marked, visible_unmarked, visible_end]
        observations = []
        for feature in features:
            observations.extend(feature)

        observations.append(on_marked_square)
        observations.append(timestep)
        return observations, action_mask

    # returns the binary representation of a direction that the agent is facing
    def get_agent_direction(self):
        return BINARY_DIRECTIONS[self.agent.direction]
        
    def get_visibility_features(self):
        mark = self.agent.tag
        visible_marked_squares = [0,0,0,0]
        visible_unmarked_squares = [0,0,0,0]
        visible_end = [0,0,0,0]
        

        for i in range(len(DELTAS)):
            for j in range(AGENT_VISION_RANGE - 1):
                x_dif, y_dif = DELTAS[i][0] * (j+1), DELTAS[i][1] * (j+1)
                new_x, new_y = self.agent.x + x_dif, self.agent.y + y_dif

                if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                    break

                new_cell = self.maze[new_y][new_x]

                if (new_x, new_y) == self.end:
                    visible_end[i] = 1

                if new_cell == mark:
                    visible_marked_squares[i] = 1

                if new_cell == 1:
                    visible_unmarked_squares[i] = 1

                if new_cell == 1:
                    break

        return visible_marked_squares, visible_unmarked_squares, visible_end

    # returns a list representing visible dead ends in all four directions
    def get_dead_ends(self):

        # binary representation of: is there a dead end in any clockwise cardinal directions, starting from north
        dead_ends = [0,0,0,0] 
        action_mask = [True, True, True, True]
        current_cell = (self.agent.x, self.agent.y)
        neighbors = self.get_path_neighbors(current_cell)

        # if there is a wall in the adjacent cell of a given direction, then that direction is a dead end
        for direction in range(len(neighbors)):
            if neighbors[direction] == False:
                dead_ends[direction] = 1
                action_mask[direction] = False

        # chooses a direction to expand vision
        for direction in range(len(DELTAS)): 
            
            # if there is a dead end in direction i, then continue to next direction
            if dead_ends[direction] == 1:
                    continue
            
            # simulates vision, by checking if next cells in the current direction are dead ends
            for j in range(AGENT_VISION_RANGE):
                next_x, next_y = self.agent.x + DELTAS[((direction+self.agent.direction)%4)][0] * (j+1), self.agent.y + DELTAS[(direction+self.agent.direction)%4][1] * (j+1)
                neighbors = self.get_path_neighbors((next_x, next_y))
                
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

        return dead_ends, action_mask

    def get_shortest_path(self):
        start = self.start
        end = self.end
        stack = [(start, [start])]
        visited = set([start])

        while stack:
            (x, y), path = stack.pop()
            if (x, y) == end:
                return path 

            for x_dif, y_dif in DELTAS: 
                next_x, next_y = x + x_dif, y + y_dif
                if (0 <= next_x < self.width and 
                    0 <= next_y < self.height and 
                    self.maze[next_y][next_x] == 0 and 
                    (next_x, next_y) not in visited):
                    visited.add((next_x, next_y))
                    stack.append(((next_x, next_y), path + [(next_x, next_y)]))

        return None  # No path found

    def display_policy(self):

        obs, mask = self.reset()
        self.set_screen()
        self.draw_maze()
        running = True
        is_moving = False
        
        def move_once():
                nonlocal obs, mask
                action, prob = self.agent.get_action(obs, mask)
                # print(f"agent took action: {ACTIONS[action % 4]}, prob: {prob}")
                obs, mask, reward, done = self.step(action)
                self.draw_maze()

                if done: 
                    obs, mask = self.reset()
                    self.set_screen()
                    self.draw_maze()  

        while running:
            current_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        obs, mask = self.reset()
                        self.set_screen()
                        self.draw_maze()
                    elif event.key == pygame.K_e:
                        move_once()

                    elif event.key == pygame.K_SPACE:
                        is_moving = True if is_moving == False else False
                                              
            if is_moving and (current_time - self.last_timestep >= TIMESTEP_LENGTH):
                self.last_timestep = current_time  
                move_once()

        pygame.quit()

if __name__ == "__main__":
    maze = Maze(rand_sizes=True)
    maze.display_policy()

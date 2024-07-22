import pygame
import random
import time
import agent

random.seed(3)
# Initialize Pygame
pygame.init()

# Colors
WHITE = pygame.Color("white") # EMPTY CELL COLOR
BLACK =  pygame.Color("black") # WALL COLOR
YELLOW = pygame.Color("gold1") # START, END, SHORTEST PATH COLOR
RED =  pygame.Color("red")
PALE_RED = (255, 100, 150)
GREEN = (0, 255, 0)

CELL_SIZE = 40
AGENT_RADIUS = CELL_SIZE/2.2
AGENT_EYE_RADIUS = AGENT_RADIUS//4

TIMESTEP_LENGTH = 0.08 # USED WHEN RENDERING THE GAME

class Maze:
    def __init__(self, max_timestep = 5000, hardcore=False, rand_start=True,
                 rand_sizes=True, rand_range=[6,12], default_size = [8,8]):

        # maze characteristics
        self.width = default_size[0] * 2 - 1
        self.height = default_size[1] * 2 - 1
        
        # these variables will be initialized upon calling self.reset() which builds the maze and sets agents
        self.layout = None
        self.start = None
        self.end = None
        self.shortest_path = None
        self.shortest_path_len = None

        self.agent = agent.Agent(RED, self)
        # self.agent = Agent(16, 10, RED, self)

        self.max_timestep = max_timestep # amount of timesteps before truncation
        self.last_timestep = time.time() # used to animate the maze

        # maze generation parameters
        self.rand_sizes = rand_sizes
        self.rand_range = rand_range
        self.rand_start = rand_start
        self.hardcore=hardcore
        self.default_size = default_size
        

    def reset(self):
        self.current_t = 0

        if self.rand_sizes == True:
            self.height = random.randint(self.rand_range[0], self.rand_range[1]) * 2 - 1
            self.width = random.randint(self.rand_range[0], self.rand_range[1]) * 2 - 1

        self.layout = [[1 for i in range(self.width)] for j in range(self.height)]
        self.build_maze()
        self.agent.reset()
        return self.agent.get_observations()
    
    def step(self, action):
        self.current_t += 1
        self.agent.total_steps+=1
        updated_estimates = False
        move,mark = action[0], action[1]
        
        # action logic
        agent_ = self.agent
        if mark == 1:
            self.layout[agent_.y][agent_.x] = agent_.tag
        direction = (move + agent_.direction) % 4
        x_dif, y_dif = agent.DELTAS[direction]
        new_x, new_y = agent_.x + x_dif, agent_.y + y_dif
        updated_estimates = agent_.move(new_x, new_y, direction)

        
        # reward function and done logic
        reward = 0
        done = False
        if (self.agent.x, self.agent.y) == self.end:
            reward = 1
            done = True
        elif updated_estimates:
            reward = 0.001      
        if self.current_t >= self.max_timestep:
            done = True
        observations, action_mask = self.agent.get_observations()
        return observations, action_mask, reward, done
    
    def is_valid_cell(self, x, y):
        return (0 <= x < self.width and 0 <= y < self.height)
    
    def set_agents(self):
        self.agent = agent.Agent(self.start[0], self.start[1], RED, self)
        self.agent.direction = 2

    def build_maze(self):
        self.set_start()
        
        # building logic starts here
        stack = [self.start]
        while stack:
            current_cell = stack[-1]
            self.layout[current_cell[1]][current_cell[0]] = 0
            neighbors = self.get_neighbors(current_cell)

            # if there are valid neighbors, we expand the maze in the direction of the randomly chosen neighbor
            if neighbors:
                next_cell = random.choice(neighbors)
                
                # removes the wall between current cell and next cell
                x1, y1 = current_cell
                x2, y2 = next_cell
                self.layout[(y1 + y2) // 2][(x1 + x2) // 2] = 0

                stack.append(next_cell)

            # no valid neighbors, therefore we backtrack until a cell with valid neighbors to continue expanding
            else:
                stack.pop()

        # if not hardcore mode, choose a random end
        if self.hardcore == False:
            self.set_end()
            self.shortest_path = self.get_shortest_path()
            self.shortest_path_len = len(self.shortest_path) - 1
            while self.shortest_path_len < 3:
                self.set_end()
                self.shortest_path = self.get_shortest_path()
                self.shortest_path_len = len(self.shortest_path) - 1
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

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []

        for x_dif, y_dif in agent.DELTAS:
            neighbor_x, neighbor_y = x + x_dif*2, y + y_dif*2

            if (self.is_valid_cell(neighbor_x, neighbor_y) and self.layout[neighbor_y][neighbor_x] == 1):
                neighbors.append((neighbor_x, neighbor_y))

        return neighbors
    
    def set_start(self):
        # if random start mode, the start is chosen randomly instead of default 0,0
        if self.rand_start == True:
            start_x = random.randint(0, (self.width - 1)//2) * 2
            start_y = random.randint(0, (self.height - 1)//2) * 2
            self.start = (start_x, start_y)
            
        else:
            self.start = (self.width//2,0) if self.width//2%2 == 0 else (self.width//2-1, 0)
            
    def set_end(self):
        coin = random.randint(0,1)
        x = 0 if coin == 0 else self.width-1
        count = 1
        while True:
            count+=1
            # print(count)
            y = random.randint(0, self.height - 1)
            if self.layout[y][x] == 0:
                self.end = (x, y)
                break
            
        # code for set end which sets in any outer edge of the maze
        
        # while True:
        #     end_x = random.randint(0, self.width - 1)
        #     end_y = random.randint(0, self.height - 1)
        #     temp_end = [end_x, end_y]
        #     coin = random.randint(0,1)
        #     end_location = self.width - 1 if coin == 0 else self.height - 1
        #     temp_end[coin] = end_location if random.randint(0,1) == 0 else 0
        #     # print(f"{temp_end} and {self.width},{self.height}")
        #     # print()
        #     if self.maze[temp_end[1]][temp_end[0]] == 0:
        #         self.end = (temp_end[0], temp_end[1])
        #         break

    def draw_maze(self):

        self.screen.fill(WHITE)
        for y in range(self.height):
            for x in range(self.width):
                if self.layout[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                if self.layout[y][x] == 2:
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

    def get_shortest_path(self):
        start = self.start
        end = self.end
        stack = [(start, [start])]
        visited = set([start])

        while stack:
            (x, y), path = stack.pop()
            if (x, y) == end:
                return path 

            for x_dif, y_dif in agent.DELTAS: 
                next_x, next_y = x + x_dif, y + y_dif
                if (self.is_valid_cell(next_x, next_y) and self.layout[next_y][next_x] == 0 and (next_x, next_y) not in visited):
                    visited.add((next_x, next_y))
                    stack.append(((next_x, next_y), path + [(next_x, next_y)]))

        return None

    def print_maze(self):
        for i in range(len(self.layout)):
            print(self.layout[i])

    def set_screen(self):
        screen_width = self.width * CELL_SIZE
        screen_height = self.height * CELL_SIZE
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Multi Agent Maze")

    def display_policy(self):

        obs, mask = self.reset()
        self.set_screen()
        self.draw_maze()
        running = True
        is_moving = False
        
        def update_env():
                
                nonlocal obs, mask
                action, prob = self.agent.get_action(obs, mask)
                # print(f"prob of action:{action} is {prob}")
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
                        update_env()
                    elif event.key == pygame.K_w:
                        self.agent.print_obs()
                    elif event.key == pygame.K_SPACE:
                        is_moving = True if is_moving == False else False
                                              
            if is_moving and (current_time - self.last_timestep >= TIMESTEP_LENGTH):
                self.last_timestep = current_time  
                update_env()

        pygame.quit()

if __name__ == "__main__":
    maze = Maze(rand_start=True, rand_sizes=True, rand_range=[15,15], hardcore=True)
    maze.display_policy()

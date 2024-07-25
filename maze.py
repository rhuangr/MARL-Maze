import pygame
import random
import time
import maze_agent

random.seed(3)
# Initialize Pygame
pygame.init()

# Colors
WHITE = pygame.Color("white") # EMPTY CELL COLOR
BLACK =  pygame.Color("black") # WALL COLOR
GREEN = pygame.Color("mediumspringgreen") # START, END, SHORTEST PATH COLOR

RED =  pygame.Color("red")
PALE_RED = pygame.Color("palevioletred1")

BLUE = pygame.Color("royalblue1")
PALE_BLUE = pygame.Color("darkslategray1")

YELLOW = pygame.Color("gold1")
PALE_YELLOW = pygame.Color("khaki1")

CELL_SIZE = 40
AGENT_RADIUS = CELL_SIZE/3
AGENT_EYE_RADIUS = AGENT_RADIUS/3
SIGNAL_RADIUS = AGENT_RADIUS * 1.2
SIGNAL_DURATION = 10

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
        self.current_t = 0

        self.agents = (maze_agent.Agent('RED', RED, PALE_RED, 2, self),
                       maze_agent.Agent('YELLOW', YELLOW, PALE_YELLOW, 3, self),
                       maze_agent.Agent('BLUE',BLUE, PALE_BLUE, 4, self))
        self.agent_positions = {}
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

        obs = []
        masks = []
        # resetting agent related
        for agent in self.agents:
            agent.reset()
            agent_obs, agent_mask = agent.get_observations()
            obs.append(agent_obs)
            masks.append(agent_mask)
        self.agent_positions[self.start] = list(self.agents)  
        return obs, masks
    
    def step(self, action):
        self.current_t += 1
        new_positions = []
        for i in range(len(self.agents)):
            agent_action = action[i]
            agent = self.agents[i]
            new_position = self.single_agent_step(agent, agent_action)
            new_positions.append((new_position,agent))

        self.agent_positions = {}
        for position, agent in new_positions:
            if position == self.end:
                agent.knows_end = 1
            if position in self.agent_positions:
                self.agent_positions[position].append(agent)
            else:
                self.agent_positions[position] = [agent]

        obs, action_masks = [], []
        for agent in self.agents:
            observation, mask = agent.get_observations()
            obs.append(observation)
            action_masks.append(mask)

        # reward function and done logic
        reward = 0
        done = False

        if len(self.agent_positions) == 1 and self.end in self.agent_positions:
            reward = 1
            done = True

        elif self.current_t > self.max_timestep:
            done = True 
        
        return obs, action_masks, reward, done
    
    def single_agent_step(self, agent, action):
        agent.current_t = self.current_t
        updated_estimates = False
        move,mark,signal = action[0], action[1], action[2]
        
        # marking
        if mark == 1:
            self.layout[agent.y][agent.x] = agent.tag
        
        # signalling
        if signal == 1:
            agent.signal_origin = (agent.x, agent.y)
            agent.is_signalling = True 
        
        if agent.signal_time <= SIGNAL_DURATION and agent.is_signalling == True:
            agent.signal_time += 1
        else:
            agent.signal_time = 0
            agent.is_signalling = False
            
        # moving
        if move != 4: # if move action chosen is not to stand still
            direction = (move + agent.direction) % 4
            x_dif, y_dif = maze_agent.DELTAS[direction]
            new_x, new_y = agent.x + x_dif, agent.y + y_dif
            updated_estimates = agent.move(new_x, new_y, direction)
            agent.memory.append(move)

        return agent.x,agent.y

    def is_valid_cell(self, x, y):
        return (0 <= x < self.width and 0 <= y < self.height)

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

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []

        for x_dif, y_dif in maze_agent.DELTAS:
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
        tags = [agent.tag for agent in self.agents]
        mark_colors = [agent.mark_color for agent in self.agents]
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.layout[y][x]
                if cell == 1:
                    pygame.draw.rect(self.screen, BLACK, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif cell in tags:
                    pygame.draw.rect(self.screen, mark_colors[tags.index(cell)], (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for x,y in self.shortest_path:
            center = self.get_cell_middle(x,y)
            pygame.draw.circle(self.screen, GREEN, center, CELL_SIZE//8)
    
        self.draw_flags() # draws flags for start and end
        self.draw_agents()
        self.draw_signal() # draws signal if any
        pygame.display.flip()

    def draw_agents(self):

        for x,y in self.agent_positions:
            agent_list = self.agent_positions[(x,y)]
            x,y = self.get_cell_middle(x,y)
            count = 0
            for agent in agent_list:
                self.draw_one_agent(agent,x,y,count,len(agent_list))
                count+=1
            
    def draw_one_agent(self,agent,x,y,count,length):
        if length == 3:
            if count == 0:
                x -= CELL_SIZE/4
                y -= CELL_SIZE/4
            elif count == 1:
                x += CELL_SIZE/4
                y -= CELL_SIZE/4
            elif count == 2:
                y += CELL_SIZE/4      
        elif length == 2:
            if count == 0:
                x -= CELL_SIZE/4
            elif count == 1:
                x += CELL_SIZE/4  
        if agent.direction == 0:
            eye_center1 = (x + CELL_SIZE//5, y - CELL_SIZE//5)
            eye_center2 = (x - CELL_SIZE//5, y - CELL_SIZE//5)
        elif agent.direction == 1:
            eye_center1 = (x + CELL_SIZE//5, y + CELL_SIZE//5)
            eye_center2 = (x + CELL_SIZE//5, y - CELL_SIZE//5)
        elif agent.direction == 2:
            eye_center1 = (x + CELL_SIZE//5, y + CELL_SIZE//5)
            eye_center2 = (x - CELL_SIZE//5, y + CELL_SIZE//5)
        elif agent.direction == 3:
            eye_center1 = (x - CELL_SIZE//5, y + CELL_SIZE//5)
            eye_center2 = (x - CELL_SIZE//5, y - CELL_SIZE//5)
        
        pygame.draw.circle(self.screen, agent.color, (x,y), AGENT_RADIUS)
        pygame.draw.circle(self.screen, BLACK, eye_center1, AGENT_EYE_RADIUS)
        pygame.draw.circle(self.screen, BLACK, eye_center2, AGENT_EYE_RADIUS)
        
    def draw_signal(self):
        for agent in self.agents:
            if agent.is_signalling == False:
                continue
            signal_center = self.get_cell_middle(agent.signal_origin[0], agent.signal_origin[1])
            outer_radius = SIGNAL_RADIUS*(agent.signal_time%4 + 1)
            color = agent.mark_color
            
            ring_surface = pygame.Surface((outer_radius * 2, outer_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(ring_surface, color, (outer_radius, outer_radius), outer_radius)
            pygame.draw.circle(ring_surface, (0, 0, 0, 0), (outer_radius, outer_radius), outer_radius-CELL_SIZE/10)
            self.screen.blit(ring_surface, (signal_center[0] - outer_radius, signal_center[1] - outer_radius))
    
    def draw_flags(self):
        end_position = self.get_cell_middle(self.end[0], self.end[1])
        start_position = self.get_cell_middle(self.start[0], self.start[1])
        flags = [end_position, start_position]

        for x,y in flags:
            # rectangle dimensions
            rect_x, rect_y = x,y - CELL_SIZE/1.2
            rect_width, rect_height = CELL_SIZE/10, CELL_SIZE/1.2

            # triangle points
            triangle_points = [
                (rect_x+rect_width, rect_y),
                (rect_x+rect_width, rect_y + rect_height//2),
                (rect_x  + rect_height//2, (rect_y + rect_y+rect_height//2)//2)
            ]
            pygame.draw.rect(self.screen, BLACK, (rect_x, rect_y, rect_width, rect_height))
            pygame.draw.polygon(self.screen, GREEN, triangle_points)
    
    def get_cell_middle(self,x,y):
        x,y = x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2
        return x,y
    
    def get_shortest_path(self):
        start = self.start
        end = self.end
        stack = [(start, [start])]
        visited = set([start])

        while stack:
            (x, y), path = stack.pop()
            if (x, y) == end:
                return path 

            for x_dif, y_dif in maze_agent.DELTAS: 
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

        obs, masks = self.reset()
        self.set_screen()
        self.draw_maze()
        running = True
        is_moving = False
        
        def update_env():
            nonlocal obs, masks
            action = []
            for i in range (len(self.agents)):
                agent_obs = obs[i]
                agent_mask = masks[i]
                agent = self.agents[i]
                agent_action, prob = agent.get_action(agent_obs, agent_mask)
                action.append(agent_action)
            # print(f"prob of action:{action} is {prob}")
            obs, masks, reward, done = self.step(action)
            self.draw_maze()

            if done: 
                obs, masks = self.reset()
                self.set_screen()
                self.draw_maze()  

        while running:
            current_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        obs, masks = self.reset()
                        self.set_screen()
                        self.draw_maze()
                    elif event.key == pygame.K_e:
                        update_env()
                    elif event.key == pygame.K_w:
                        for i in range(len(self.agents)):
                            self.agents[i].print_obs(obs[i])
                    elif event.key == pygame.K_SPACE:
                        is_moving = True if is_moving == False else False
                                            
            if is_moving and (current_time - self.last_timestep >= TIMESTEP_LENGTH):
                self.last_timestep = current_time  
                update_env()

        pygame.quit()

if __name__ == "__main__":
    maze = Maze(rand_start=False, rand_sizes=True, rand_range=[2,2], hardcore=False)
    maze.display_policy()

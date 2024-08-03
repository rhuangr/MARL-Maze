import pygame
import random
import time


PATH_COLOR = pygame.Color("white")
WALL_COLOR =  pygame.Color("black")
FLAG_COLOR = pygame.Color("mediumspringgreen") 
KEY_COLOR = pygame.Color('darkgoldenrod1')

CELL_SIZE = 40
AGENT_RADIUS = CELL_SIZE/3
AGENT_EYE_RADIUS = AGENT_RADIUS/3
SIGNAL_RADIUS = AGENT_RADIUS * 1.2
SIGNAL_DURATION = 6

TIMESTEP_LENGTH = 0.08 # USED WHEN RENDERING THE GAME
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # change in x,y after moving in respective cardinal direction

class Maze:
    def __init__(self, agents, max_timestep = 3500, difficulty=1, rand_start=False,
                 rand_sizes=False, rand_range=[6,12], default_size = [8,8]):

        # maze characteristics
        self.width = default_size[0] * 2 - 1
        self.height = default_size[1] * 2 - 1
        
        # these variables will be initialized upon calling self.reset() which builds the maze and sets agents
        self.layout = None
        self.start = None
        self.end = None
        self.key = None
        self.shortest_path = None
        self.shortest_path_len = None
        self.exit_found = False
        self.current_t = 0

        self.agents = agents
        for agent in self.agents:
            agent.maze = self
            agent.brain.maze = self
        self.agent_positions = {}
        # self.agent = Agent(16, 10, RED, self)

        self.max_timestep = max_timestep # amount of timesteps before truncation
        self.last_timestep = time.time() # used to animate the maze

        # maze generation parameters
        self.rand_sizes = rand_sizes
        self.rand_range = rand_range
        self.rand_start = rand_start
        self.difficulty=difficulty
        self.default_size = default_size
        

    def reset(self):
        self.current_t = 0
        self.build_maze()

        self.agent_positions = {}
        self.agent_positions[self.start] = list(self.agents)
        self.exit_found = False

        obs = []
        masks = []
        # resetting agent related
        for agent in self.agents:
            agent.reset(self.start[0], self.start[1])
            agent_obs, agent_mask = agent.get_observations()
            obs.append(agent_obs)
            masks.append(agent_mask)

        return obs, masks
    
    def step(self, action):
        self.current_t += 1
        new_positions = []
        total_updates = 0
        total_visits = []
        agents_have_key = False
        first_exit_find = False
        first_key_find = False
        for i in range(len(self.agents)):
            agent_action = action[i]
            agent = self.agents[i]
            new_x, new_y, updated_estimates, has_key, got_key, cell_visits = self.single_agent_step(agent, agent_action)
            # all_to_exit = all_to_exit and to_exit
            new_position = (new_x, new_y)
            total_updates += updated_estimates
            total_visits.append(cell_visits)
            agents_have_key = agents_have_key or has_key
            first_key_find = first_key_find or got_key
            new_positions.append((new_position,agent))

        self.agent_positions = {}
        for position, agent in new_positions:
            if position in self.agent_positions:
                self.agent_positions[position].append(agent)
            else:
                self.agent_positions[position] = [agent]
                
        obs, action_masks = [], []
        for agent in self.agents:
            observation, mask = agent.get_observations()
            obs.append(observation)
            action_masks.append(mask)
            if self.exit_found == False and agent.knows_end:
                first_exit_find = agent.knows_end
                self.exit_found = True
        
        # reward function and done logic
        reward = 0 + first_key_find*0.5 + first_exit_find*0.5#+ total_updates*0.001 + all_to_exit*0.004
        for visit in total_visits:
            reward = reward + (visit-3)*-0.0001 if visit > 3 else reward
        done = False
        if agents_have_key and len(self.agent_positions) == 1 and self.end in self.agent_positions :
            reward = 1
            done = True
        elif self.current_t >= self.max_timestep:
            done = True 
        
        return obs, action_masks, reward, done
    
    def single_agent_step(self, agent, action):
        
        agent.current_t = self.current_t
        updated_estimates = False
        got_key = False
        move,mark = action[0], action[1]
        # print(f"{agent.name}: Move: {move}")
        # marking
        if mark == 1:
            self.layout[agent.y][agent.x] = agent.tag
        
        # signalling
        # if signal == 1:
        #     agent.signal_origin = (agent.x, agent.y)
        #     agent.is_signalling = True 
        
        if agent.signal_time <= SIGNAL_DURATION and agent.is_signalling == True:
            agent.signal_time += 1
        else:
            agent.signal_time = 0
            agent.is_signalling = False
            
        # moving
        if move != 4: # if move action chosen is not to stand still
            direction = (move + agent.direction) % 4
            x_dif, y_dif = DELTAS[direction]
            new_x, new_y = agent.x + x_dif, agent.y + y_dif
            agent.move(new_x, new_y, direction)
            if (new_x, new_y) == self.key:
                self.key = 0
                agent.has_key = True
                agent.team_has_key = True
                got_key = True
            agent.memory.append(move)
            visits = agent.update_visited_cells() # returns the amount of times the agent has been on that cell, used for rew
            
            # print(f"name: {agent.name}, pos: {agent.x}, {agent.y} visits: {visits}")

        return agent.x,agent.y,updated_estimates,agent.has_key, got_key, visits #to_exit

    def is_valid_cell(self, x, y):
        return (0 <= x < self.width and 0 <= y < self.height)

    def build_maze(self):
        if self.rand_sizes == True:
            size = random.randint(self.rand_range[0], self.rand_range[1]) * 2 - 1
            self.height = size
            self.width = size

        self.layout = [[1 for i in range(self.width)] for j in range(self.height)]
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

        # choose end n amount of times equals to difficulty constant and choose the longest path
        max_length = 0
        lengths_ends = {}
        lengths_paths = {}
        for _ in range(self.difficulty):
            self.set_end()
            shortest_path = self.get_shortest_path(self.start, self.end)
            shortest_path_len = len(shortest_path)
            max_length = max(max_length, shortest_path_len)
            lengths_ends[shortest_path_len] = self.end
            lengths_paths[shortest_path_len] = shortest_path

        self.end = lengths_ends[max_length]
        self.shortest_path = lengths_paths[max_length]
        self.shortest_path_len = max_length
        self.set_key()

    def get_neighbors(self, cell):
        x, y = cell
        neighbors = []

        for x_dif, y_dif in DELTAS:
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
        while True:
            y = random.randint(0, self.height - 1)
            if (x,y) == self.start:
                continue
            elif self.layout[y][x] == 0:
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
        
    def set_key(self):
        min_path = max(self.height, self.width) #min(self.height, self.width) * (max(self.height, self.width)/)
        while True:
            x,y = random.randint(0,self.width-1), random.randint(0,self.height-1)
            if (self.layout[y][x] == 1 or (x,y) == self.end or (x,y) == self.start or 
                len(self.get_shortest_path(self.end, (x,y))) < min_path or 
                len(self.get_shortest_path(self.start, (x,y))) < min_path):
                continue
            self.key = (x,y)
            self.layout[y][x] = 5
            break

    def get_shortest_path(self, start, end):
        stack = [(start, [start])]
        visited = set([start])

        while stack:
            (x, y), path = stack.pop()
            if (x, y) == end:
                return path 

            for x_dif, y_dif in DELTAS: 
                next_x, next_y = x + x_dif, y + y_dif
                if (self.is_valid_cell(next_x, next_y) and self.layout[next_y][next_x] == 0 and (next_x, next_y) not in visited):
                    visited.add((next_x, next_y))
                    stack.append(((next_x, next_y), path + [(next_x, next_y)]))

        return None


    # EVERYTHING BELOW RELATES ONLY TO RENDERING THE MAZE.
    def draw_maze(self):
        self.screen.fill(PATH_COLOR)
        tags = [agent.tag for agent in self.agents]
        mark_colors = [agent.mark_color for agent in self.agents]
        
        for y in range(self.height):
            for x in range(self.width):
                cell = self.layout[y][x]
                if cell == 1:
                    pygame.draw.rect(self.screen, WALL_COLOR, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif cell in tags:
                    pygame.draw.rect(self.screen, mark_colors[tags.index(cell)], (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for x,y in self.shortest_path:
            center = self.get_cell_middle(x,y)
            pygame.draw.circle(self.screen, FLAG_COLOR, center, CELL_SIZE//8)
    
        self.draw_flags() # draws flags for start and end
        self.draw_agents()
        if self.key != 0:
            self.draw_key()
        # self.draw_signal() # draws signal if any
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
        if agent.has_key:
            pygame.draw.circle(self.screen, KEY_COLOR, (x,y), CELL_SIZE/2.3)
        pygame.draw.circle(self.screen, agent.color, (x,y), AGENT_RADIUS)
        pygame.draw.circle(self.screen, WALL_COLOR, eye_center1, AGENT_EYE_RADIUS)
        pygame.draw.circle(self.screen, WALL_COLOR, eye_center2, AGENT_EYE_RADIUS)
        
    def draw_signal(self):
        for agent in self.agents:
            if agent.is_signalling == False:
                continue
            signal_center = self.get_cell_middle(agent.signal_origin[0], agent.signal_origin[1])
            outer_radius = SIGNAL_RADIUS*(agent.signal_time%5 + 1)
            color = agent.color

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
            pygame.draw.rect(self.screen, WALL_COLOR, (rect_x, rect_y, rect_width, rect_height))
            pygame.draw.polygon(self.screen, FLAG_COLOR, triangle_points)

    def draw_key(self):
        key_x, key_y = self.get_cell_middle(self.key[0], self.key[1])
        key_y -= CELL_SIZE/4
        key_color = KEY_COLOR
        pygame.draw.circle(self.screen, key_color, (key_x, key_y), CELL_SIZE/6)
        pygame.draw.circle(self.screen, PATH_COLOR, (key_x, key_y), CELL_SIZE/11)
        pygame.draw.rect(self.screen, key_color, (self.key[0]*CELL_SIZE+CELL_SIZE/2.2, self.key[1]*CELL_SIZE+CELL_SIZE/3, CELL_SIZE/10, CELL_SIZE/2))
        pygame.draw.rect(self.screen, key_color, (self.key[0]*CELL_SIZE+CELL_SIZE/2.2, self.key[1]*CELL_SIZE+CELL_SIZE*2/3, CELL_SIZE/4.5, CELL_SIZE/10))
        pygame.draw.rect(self.screen, key_color, (self.key[0]*CELL_SIZE+CELL_SIZE/2.2, self.key[1]*CELL_SIZE+CELL_SIZE/2, CELL_SIZE/4.5, CELL_SIZE/10))

    def get_cell_middle(self,x,y):
        x,y = x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2
        return x,y

    def print_maze(self):
        for i in range(len(self.layout)):
            print(self.layout[i])

    def set_screen(self):
        screen_width = self.width * CELL_SIZE
        screen_height = self.height * CELL_SIZE
        self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        pygame.display.set_caption("Multi Agent Maze")

    def display_policy(self):
        pygame.init()
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
                        print(self.agent_positions)
                        for i in range(len(self.agents)):
                            self.agents[i].print_obs(obs[i])
                    elif event.key == pygame.K_SPACE:
                        is_moving = True if is_moving == False else False
                                            
            if is_moving and (current_time - self.last_timestep >= TIMESTEP_LENGTH):
                self.last_timestep = current_time  
                update_env()

        pygame.quit()
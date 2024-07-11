import pygame
import random
import time
from agent import Agent

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255) # EMPTY CELL COLOR
BLACK = (0, 0, 0) # WALL COLOR
RED = (255, 150, 150)
GREEN = (0, 255, 0)

CELL_SIZE = 20
AGENT_RADIUS = CELL_SIZE/2
AGENT_EYE_RADIUS = AGENT_RADIUS//4

TIMESTEP_LENGTH = 0.5 # ONE TIME STEP LASTS 0.5 SECONDS

# TUPLES REPRESENTING THE CHANGE IN X,Y OF ANY CELL GIVEN THE DIRECTION THAT THE AGENT IS FACING AND THE POSSIBLE ACTIONS:
# ['move forward', 'backward', 'left', 'right']
FACING_NORTH_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FACING_SOUTH_DELTA = [(1, 0), (1, 0), (0, -1), (0, 1)]
FACING_WEST_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FACING_EAST_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class Maze:
    def __init__(self, width, height):
        self.width = width*2 -1
        self.height = height*2 -1
        self.maze = None
        self.start = (0, 0)
        self.end = None
        self.agent = Agent(5, 5, RED)
        self.last_timestep = time.time()

        self.shortest_path = []
        self.shortest_path_len = 0
        self.set_screen()

    # resets the data structure for the maze, where 1 represents walls and 0 represents paths
    def reset_maze(self):
        self.maze = [[1 for i in range(self.width)] for j in range(self.height)]

    def generate_maze(self):
        self.reset_maze()
        stack = [self.start]
        while stack:
            current = stack[-1]
            self.maze[current[1]][current[0]] = 0

            neighbors = self.get_neighbors(current)

            # if there are valid neighbors, we expand the maze in the direction of the randomly chosen neighbor
            if neighbors:
                next_cell = random.choice(neighbors)
                self.remove_wall(current, next_cell)
                stack.append(next_cell)

            # no valid neighbors, therefore we backtrack until a cell with valid neighbors to continue expanding
            else:
                stack.pop()

        self.set_end()
        self.shortest_path = self.get_shortest_path()
        self.shortest_path_len = len(self.shortest_path) - 1
        self.draw_maze()
        self.draw_agent()
        print(self.shortest_path_len)

    def get_neighbors(self, cell, generate=True):
        x, y = cell
        neighbors = []
        for x_dif, y_dif in FACING_NORTH_DELTA:
            if generate == True:
                neighbor_x, neighbor_y = x + x_dif*2, y + y_dif*2
            else:
                neighbor_x, neighbor_y = x + x_dif, y + y_dif
            if 0 <= neighbor_x < self.width and 0 <= neighbor_y < self.height and self.maze[neighbor_y][neighbor_x] == 1:
                neighbors.append((neighbor_x, neighbor_y))
        return neighbors

    def remove_wall(self, cell1, cell2):
        x1, y1 = cell1
        x2, y2 = cell2
        self.maze[(y1 + y2) // 2][(x1 + x2) // 2] = 0

    def set_end(self):
        end = random.randint(0,1)
        if end == 0:
            y = self.height - 1
            while True:
                x = random.randint(0, self.width - 1)
                if self.maze[y][x] == 0:
                    self.end = (x, y)
                    break
        else:
            x = self.width - 1
            while True:
                y = random.randint(0, self.height - 1)
                if self.maze[y][x] == 0:
                    self.end = (x, y)
                    break
        

    def draw_maze(self):
        self.screen.fill(WHITE)
        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 1:
                    pygame.draw.rect(self.screen, BLACK, (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for x,y in self.shortest_path:
            pygame.draw.rect(self.screen, (120,120,120), (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw start
        pygame.draw.rect(self.screen, GREEN, (self.start[0]*CELL_SIZE, self.start[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        # Draw end
        pygame.draw.rect(self.screen, RED, (self.end[0]*CELL_SIZE, self.end[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()

    def draw_agent(self):

        x = self.agent.x * CELL_SIZE + CELL_SIZE//2
        y = self.agent.y * CELL_SIZE + CELL_SIZE//2
        agent_center = (x,y)

        eye_center1 = (x - CELL_SIZE//5, y + CELL_SIZE//5)
        eye_center2 = (x + CELL_SIZE//5, y + CELL_SIZE//5)

        pygame.draw.circle(self.screen, self.agent.color, agent_center, AGENT_RADIUS)
        pygame.draw.circle(self.screen, BLACK, eye_center1, AGENT_EYE_RADIUS)
        pygame.draw.circle(self.screen, BLACK, eye_center2, AGENT_EYE_RADIUS)
        pygame.display.flip()

    def set_screen(self):
        screen_width = self.width * CELL_SIZE
        screen_height = self.height * CELL_SIZE
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Random Maze Generator")

    def step(self, action):
        if action > 3:
            action = action - 4
            self.maze[self.agent.y][self.agent.x] = -1

        if action == 0 and self.agent.y - 1 >= 0: # up
            self.agent.y -= 1
        elif action == 1 and self.agent.y + 1 < self.height : # down
            self.agent.y += 1
        elif action == 2 and self.agent.x - 1 >= 0: # left
            self.agent.x -= 1
        elif action == 3 and self.agent.x + 1 < self.width: # right
            self.agent.x += 1
        
        reward = 0 if (self.agent.x, self.agent.y) == self.end else -1

        return
    
    def get_observations(self):
        observations = []
        
        pass
    
    def get_dead_end(self):
        pass
    
    def get_shortest_path(self):
        start = self.start
        end = self.end
        stack = [(start, [start])]
        visited = set([start])

        while stack:
            (x, y), path = stack.pop()
            if (x, y) == end:
                return path 

            for x_dif, y_dif in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # down, right, up, left
                next_x, next_y = x + x_dif, y + y_dif
                if (0 <= next_x < self.width and 
                    0 <= next_y < self.height and 
                    self.maze[next_y][next_x] == 0 and 
                    (next_x, next_y) not in visited):
                    visited.add((next_x, next_y))
                    stack.append(((next_x, next_y), path + [(next_x, next_y)]))

        return None  # No path found


def main():
    maze_width, maze_height = 12,12
    
    # Generate and draw the maze
    maze_gen = Maze(maze_width, maze_height)
    maze_gen.generate_maze()

    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Generate a new maze when space is pressed
                    maze_gen.generate_maze()
        
        # if time.time() - maze_gen.last_timestep >= TIMESTEP_LENGTH:
        #     maze_gen.last_timestep = time.time()
        #     maze_gen.agent.x += 1
        #     maze_gen.agent.y += 1
        # maze_gen.draw_maze()

    pygame.quit()

if __name__ == "__main__":
    main()
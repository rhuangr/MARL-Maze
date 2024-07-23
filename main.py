# from maze_agent import Agent
# from maze import Maze
# from collections import deque

# def main():
#     maze = Maze(rand_range=[10,14], rand_start=False, hardcore=True)
#     maze.agents.brain.train()
# main()
# import pygame

# # Initialize Pygame
# pygame.init()

# # Set up display
# width, height = 800, 600
# window = pygame.display.set_mode((width, height))
# pygame.display.set_caption("Simple Flag")

# # Define colors
# background_color = (255, 255, 255)  # White background
# rectangle_color = (0, 0, 255)       # Blue rectangle
# triangle_color = (255, 0, 0)        # Red triangle

# # Define rectangle dimensions
# rect_x, rect_y = 100, 100
# rect_width, rect_height = 600, 400

# # Define triangle points
# triangle_points = [
#     (rect_x, rect_y),  # Top-left of the rectangle
#     (rect_x, rect_y + rect_height),  # Bottom-left of the rectangle
#     (rect_x + rect_width // 2, rect_y + rect_height // 2)  # Middle of the right side of the rectangle
# ]

# # Run until the user asks to quit
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Fill the background
#     window.fill(background_color)

#     # Draw the rectangle
#     pygame.draw.rect(window, rectangle_color, (rect_x, rect_y, rect_width, rect_height))

#     # Draw the triangle
#     pygame.draw.polygon(window, triangle_color, triangle_points)

#     # Update the display
#     pygame.display.flip()

# # Quit Pygame
# pygame.quit()

x = set([1,3,2,1,2])
print(x)
print(len(x))
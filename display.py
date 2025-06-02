import pygame
import sys
import numpy as np

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
POINT_COLOR = (255, 255, 255)  # White
BG_COLOR = (0, 0, 0)           # Black
POINT_RADIUS = 2              # Pixel radius of points

# World coordinate bounds
WORLD_MIN = -500
WORLD_MAX = 500

# Global screen variable
screen = None

def init():
    """
    Initialize the Pygame window and rendering context.
    """
    global screen
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Coordinate Visualization")
    screen.fill(BG_COLOR)
    pygame.display.flip()

def update(coords):
    """
    Update the Pygame window with the given coordinates.

    Args:
        coords (list of np.ndarray or np.ndarray): 3D coordinates (x, y, z); only x and y are used.
    """
    global screen

    # Event handling (allows window closing)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(BG_COLOR)

    # Ensure coords is a NumPy array and extract (x, y)
    coords_array = np.array(coords)
    coords_xy = coords_array[:, :2]

    for x, y in coords_xy:
        screen_x = int((x - WORLD_MIN) / (WORLD_MAX - WORLD_MIN) * WINDOW_WIDTH)
        screen_y = int((1 - (y - WORLD_MIN) / (WORLD_MAX - WORLD_MIN)) * WINDOW_HEIGHT)
        pygame.draw.circle(screen, POINT_COLOR, (screen_x, screen_y), POINT_RADIUS)

    pygame.display.flip()

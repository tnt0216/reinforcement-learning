"""
The code within this script is a simple 2x2 Grid-World Implementation using the Q-learning algorithm
(the most basic RL algorithm). This implementation is exploritory in nature and will include variations
of the same problem to investigate the following basic RL concepts:
    1) Deterministic vs Stochastic Transitions
    2) Convergence Behavior
    3) Exploration Stategies
    4) Tabular vs Function Approximators
    5) Impacts of Reward Shaping

Background of Problem: You live in a 4-room house and just bought a roomba to deal with some of the 
daily dirt that has been collecting while you are busy working during the week. The roomba is only 
able to perform 5 actions: move - up, down, right, left, and clean. Each time the roomba successfully 
cleans a dirty room it gets rewarded +10, however, if it cleans a clean room it is penalized -1. 
Moving up, down, right, or left into another room results in no reward or penalization. 

Givens:
    States (S): A, B, C, D (4 total states - one for each room)
    Actions (A): up, down, right, left, clean
    Rewards (R): clean dirty = +10, clean clean = -1, move = 0
    Learning Rate (α): 0.5
    Discount Factor (γ): 0.9
"""

# Imported Packages
import pygame


def drawGrid(screen, screen_width, screen_height, state_conditions):
    """
    This helper function draws the grid, updates visuals, and renders the state space the agent 
    interacts with.
    """

    # Constants
    black = (0, 0, 0)
    white = (200, 200, 200)
    pygame.font.init()  # Need to add font to screen to signal state condition
    my_font = pygame.font.SysFont("Arial", 24)
    block_size = screen_width // 2 # Dividing the screen into a 2x2 (Divides cells per pixel)

    # Dictionary to establish text placement for state conditions on gridworld
    cell_positions = {
        "A": (0, 0),
        "B": (block_size, 0),
        "C": (0, block_size),
        "D": (block_size, block_size)
    }

    screen.fill(white)

    # This loop creates the individual cells using the Rect pygame object - stores rectangular coordinates
    for x in range(0, screen_width, block_size):
        for y in range(0, screen_height, block_size):
            rect = pygame.Rect(x, y, block_size, block_size)
            pygame.draw.rect(screen, black, rect, 1)

    text_surfs = {}

    # This loop renders the state conditions
    for key, value in state_conditions.items():
        surf = my_font.render(str(value), True, black)
        rect = surf.get_rect()

        x, y = cell_positions[key]
        rect.center = (x + block_size // 2, y + block_size // 2)

        text_surfs[key] = (surf, rect)

    return text_surfs


def main():

    # Defining Constants - RL agent
    alpha = 0.5  # Learning rate 
    gamma = 0.9  # Discount factor 
    actions = ["up", "down", "right", "left", "clean"]  # Actions that the agent can take within the environment
    
    # Initial state conditions for each cell shown in the Pygame rendering
    state_conditions = {
        "A": "dirty",
        "B": "clean",
        "C": "dirty",
        "D": "clean"
    }

    # Defining Constants - PyGame Rendering
    screen_width = 500
    screen_height = 500

    # Initializing a simple 2x2 grid in Pygame for visualization
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Q-Learning - 4 Room House Problem")

    # Looping to update using the drawGrid helper function
    running = True
    while running:
        text_surfs = drawGrid(screen, screen_width, screen_height, state_conditions)

        for surf, rect in text_surfs.values():
            screen.blit(surf, rect)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


if __name__ == "__main__":
    main()


        
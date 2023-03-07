import pygame

# Initialize Pygame and set the window size
pygame.init()
size = (700, 500)
screen = pygame.display.set_mode(size)

# Set the background color for the empty grid spaces
bg_color = (255, 255, 255)
screen.fill(bg_color)

# Create entities using rectangles
food_color = (0, 255, 0)
food = pygame.draw.rect(screen, food_color, (10, 10, 20, 20))

water_color = (0, 0, 255)
water = pygame.draw.rect(screen, water_color, (30, 30, 20, 20))

poison_color = (255, 0, 0)
poison = pygame.draw.rect(screen, poison_color, (50, 50, 20, 20))

# Create agents using circles
agent_color = (255, 255, 0)
agent = pygame.draw.circle(screen, agent_color, (75, 75), 10)

# Update the display
pygame.display.flip()

# Handle user input
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
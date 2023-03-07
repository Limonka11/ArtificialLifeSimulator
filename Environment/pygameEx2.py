import pygame
import random

#from ..entities.DQNBrain import DQN

# Initialize Pygame
pygame.init()

# Set the size of the environment
size = (700, 500)
screen = pygame.display.set_mode(size)

# Load the forest background image
background_image = pygame.image.load("forest.jpg").convert()

# Create a class for the food entity
class Food:
    def __init__(self, x, y, nutrition):
        self.x = x
        self.y = y
        self.nutrition = nutrition
        self.image = pygame.image.load("food.png").convert_alpha()

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

# Create a class for the agent entity
class Agent:
    def __init__(self, x, y, health, hunger, thirst, age, brain):
        self.x = x
        self.y = y
        self.health = health
        self.hunger = hunger
        self.thirst = thirst
        self.age = age
        self.brain = brain
        self.image = pygame.image.load("agent.png").convert_alpha()

    def draw(self):
        screen.blit(self.image, (self.x, self.y))


# Create a class for the water entity
class Water:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.image.load("water.png").convert_alpha()

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

# Create a class for the poison entity
class Poison:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.image = pygame.image.load("poison.png").convert_alpha()

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

# Create a list to store all the entities
entities = []

# Add some random food, agent, water and poison entities to the list
for i in range(5):
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])
    nutrition = random.randint(10, 50)
    entities.append(Food(x, y, nutrition))

for i in range(5):
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])
    health = random.randint(50, 100)
    hunger = random.randint(0, 10)
    thirst = random.randint(0, 10)
    age = random.randint(0, 10)
    brain = 1 #DQN(state_size, action_size)
    entities.append(Agent(x, y, health, hunger, thirst, age, brain))

for i in range(5):
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])
    entities.append(Water(x, y))

for i in range(5):
    x = random.randint(0, size[0])
    y = random.randint(0, size[1])
    entities.append(Poison(x, y))
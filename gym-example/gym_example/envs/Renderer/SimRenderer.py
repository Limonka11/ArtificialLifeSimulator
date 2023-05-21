import random
import numpy as np
import pygame as pg
from typing import List
from ..entities import Entity, Agent, EntityTypes, Actions, Wolf

class SimRenderer:
    def __init__(self,
                width: int,
                height: int,
                square_size: int):

        self.width = width
        self.height = height
        self.square_size = square_size
        self.entities = EntityTypes

        self.colors = [
            (24, 255, 255),  # Light Blue
            (255, 238, 88),  # Yellow
            (255, 94, 89),  # Red
            (255, 255, 255),  # White
            (126, 87, 194),  # Purple
            (66, 165, 245),  # Light Blue
            (121, 85, 72),  # Brown
            (0, 200, 83),  # Green
        ]

        # Pygame related vars
        self.background = None
        self.clock = None
        self.screen = None
        self.rendered = False

        water_colors = {i: {j: None} for i in range(self.width) for j in range(self.height)}
        for i in range(self.width):
            for j in range(self.height):
                offset_r = 0
                offset_g = 0
                offset_b = 0
                if random.random() > (1-.9):
                    offset_r = random.randint(0, 30)

                if random.random() > (1-.9):
                    offset_g = random.randint(0, 30)

                if random.random() > (1-.9):
                    offset_b = random.randint(0, 30)

                water_colors[i][j] = (0 + offset_r, 255 - offset_g, 255 - offset_b)
        
        self.water_colors = water_colors

    def render(self, agents: List[Agent], grid: np.array, fps: int) -> bool:

        # Initialize pygame
        if not self.rendered:
            pg.init()
            self.screen = pg.display.set_mode((round(self.width) * self.square_size, round(self.height) * self.square_size))
            self.clock = pg.time.Clock()
            self.clock.tick(fps)

            # Background
            self.background = pg.Surface((round(self.width) * self.square_size, round(self.height) * self.square_size))
            self.draw_tiles()

            # Create a surface object, image is drawn on it.
            heart = pg.image.load("C:\\imperial\\MengProject\\Assets\\reproduce.png").convert_alpha()
            self.scaled_heart = pg.transform.scale(heart, (15, 15))

            sword = pg.image.load("C:\\imperial\\MengProject\\Assets\\attack.png").convert_alpha()
            self.scaled_sword = pg.transform.scale(sword, (15, 15))

            tree = pg.image.load("C:\\imperial\\MengProject\\Assets\\tree.png").convert_alpha()
            self.scaled_tree = pg.transform.scale(tree, (25, 25))

            self.rendered = True

        # Draw and update all entities
        self.screen.blit(self.background, (0, 0))
        
        self.draw_agents(agents)
        self.draw_food(grid)
        self.draw_pheromoes(grid)
        self.draw_trees(grid)

        pg.display.update()
        self.clock.tick(fps)

        return self.check_pygame_exit()

    def draw_agents(self, agents: List[Agent]):
        for agent in agents:
            if not agent.dead:
                if agent.gene < len(self.colors):
                    body_color = self.colors[agent.gene]
                else:
                    body_color = self.colors[agent.gene % len(self.colors)]

                # Draw the bodies of agents
                j = (agent.j * self.square_size) + max(1, int(self.square_size / 8))
                i = (agent.i * self.square_size) + max(1, int(self.square_size / 8))
                size = self.square_size - max(1, int(self.square_size / 8) * 2)
                surface = (j, i, size, size)
                
                if type(agent) == Agent:
                    pg.draw.circle(self.screen, body_color, (j+size/2, i+size/2), size/2)
                elif type(agent) == Wolf:
                    pg.draw.rect(self.screen, body_color, pg.Rect(j, i, size, size))
                
                # draw the eyes of egents
                size = self.square_size - max(1, int(self.square_size * .9))

                j = (agent.j * self.square_size) + max(1, int(self.square_size / 3))
                i = (agent.i * self.square_size) + max(1, int(self.square_size / 3))
                surface = (j, i, size, size)
                pg.draw.rect(self.screen, (0, 0, 0), surface, 0)

                j = (agent.j * self.square_size) + max(1, int(self.square_size / 1.8))
                i = (agent.i * self.square_size) + max(1, int(self.square_size / 3))
                surface = (j, i, size, size)
                pg.draw.rect(self.screen, (0, 0, 0), surface, 0)

                self.draw_agent_heart(agent)
                self.draw_sword(agent)

    def draw_food(self, grid: np.array):
        food = grid.get_entities(self.entities.food)
        for item in food:
            self.draw_rect(item, color=(255, 255, 255))

        corpse = grid.get_entities(self.entities.corpse)
        for item in corpse:
            self.draw_rect(item, color=(255, 0, 0))

        poison = grid.get_entities(self.entities.poison)
        for item in poison:
            self.draw_rect(item, color=(0, 0, 0))

        water = grid.get_entities(self.entities.water)
        for item in water:
            self.draw_water_rect(item)

    def draw_trees(self, grid: np.array):
        trees = grid.get_entities(self.entities.tree)

        for tree in trees:
            j = (tree.j * self.square_size) + max(1, int(self.square_size / 8))
            i = (tree.i * self.square_size) + max(1, int(self.square_size / 8))
            size = self.square_size - int(self.square_size / 2.5) * 2
            self.screen.blit(self.scaled_tree, (j - size, i - size))
    
    def draw_pheromoes(self, grid: np.array):
        pheromones = grid.get_entities(self.entities.pheromone)
        for item in pheromones:
            self.draw_rect(item, color=(126, 87, 194))

    def draw_rect(self, item: Entity, color: tuple):
        j = (item.j * self.square_size) + int(self.square_size / 2.5)
        i = (item.i * self.square_size) + int(self.square_size / 2.5)
        size = self.square_size - int(self.square_size / 2.5) * 2
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, color, surface, 0)

    def draw_agent_heart(self, agent: Agent):
        if agent.action == Actions.reproduce:
            # Get the coordinates
            j = (agent.j * self.square_size) + max(1, int(self.square_size / 8))
            i = (agent.i * self.square_size) + max(1, int(self.square_size / 8))
            size = self.square_size - max(1, int(self.square_size / 8) * 2)
            self.screen.blit(self.scaled_heart, (j+size/16-2.5, i+size/6))

    def draw_sword(self, agent: Agent):
        if agent.action in [Actions.attack_down, Actions.attack_left, Actions.attack_right, Actions.attack_up]:
            # Get the coordinates
            j = (agent.j * self.square_size) + max(1, int(self.square_size / 8))
            i = (agent.i * self.square_size) + max(1, int(self.square_size / 8))
            size = self.square_size - max(1, int(self.square_size / 8) * 2)
            self.screen.blit(self.scaled_sword, (j+size/16-2.5, i - 5))

    def draw_water_rect(self, item: Entity):
        j = (item.j * self.square_size) 
        i = (item.i * self.square_size)
        surface = (j , i, self.square_size, self.square_size)
        pg.draw.rect(self.screen, self.water_colors[item.i][item.j], surface, 0)

    def draw_tiles(self):
        tile_colors = {i: {j: None} for i in range(self.width) for j in range(self.height)}
        for i in range(self.width):
            for j in range(self.height):
                offset_r = 0
                offset_g = 0
                offset_b = 0
                if random.random() > (1-.9):
                    offset_r = random.randint(-30, 30)

                tile_colors[i][j] = (50 + offset_r, 205 + offset_g, 50 + offset_b)

        for i in range(self.width):
            for j in range(self.height):
                pg.draw.rect(self.background, tile_colors[i][j],
                             (i * self.square_size, j * self.square_size, self.square_size, self.square_size), 0)

    def check_pygame_exit(self) -> bool:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return False
        return True
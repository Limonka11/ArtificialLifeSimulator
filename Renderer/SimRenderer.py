import random
import numpy as np
import pygame as pg
from typing import List
import sys
sys.path.append("C:\\imperial\\MengProject\\Environment")
from entities import Entity, Agent, EntityTypes

class SimRenderer:
    def __init__(self,
                width: int,
                height: int,
                grid_size: int):

        self.width = width
        self.height = height
        self.grid_size = grid_size
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
            self.screen = pg.display.set_mode((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
            self.clock = pg.time.Clock()
            self.clock.tick(fps)

            # Background
            self.background = pg.Surface((round(self.width) * self.grid_size, round(self.height) * self.grid_size))
            self.draw_tiles()
            self.rendered = True

        # Draw and update all entities
        self.screen.blit(self.background, (0, 0))
        self.draw_agents(agents)
        self.draw_food(grid)
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
                j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 8))
                i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 8))
                size = self.grid_size - max(1, int(self.grid_size / 8) * 2)
                surface = (j, i, size, size)
                pg.draw.circle(self.screen, body_color, (j+size/2, i+size/2), size/2)
                
                size = self.grid_size - max(1, int(self.grid_size * .9))

                # draw the eyes of egents
                j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 3))
                i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 3))
                surface = (j, i, size, size)
                pg.draw.rect(self.screen, (0, 0, 0), surface, 0)

                j = (agent.j * self.grid_size) + max(1, int(self.grid_size / 1.8))
                i = (agent.i * self.grid_size) + max(1, int(self.grid_size / 3))
                surface = (j, i, size, size)
                pg.draw.rect(self.screen, (0, 0, 0), surface, 0)

    def draw_food(self, grid: np.array):
        food = grid.get_entities(self.entities.food)
        for item in food:
            self.draw_food_rect(item, color=(255, 255, 255))

        poison = grid.get_entities(self.entities.poison)
        for item in poison:
            self.draw_food_rect(item, color=(0, 0, 0))

        water = grid.get_entities(self.entities.water)
        for item in water:
            self.draw_water_rect(item)
    
    def draw_food_rect(self, item: Entity, color: tuple):
        j = (item.j * self.grid_size) + int(self.grid_size / 2.5)
        i = (item.i * self.grid_size) + int(self.grid_size / 2.5)
        size = self.grid_size - int(self.grid_size / 2.5) * 2
        surface = (j, i, size, size)
        pg.draw.rect(self.screen, color, surface, 0)

    def draw_water_rect(self, item: Entity):
        j = (item.j * self.grid_size) 
        i = (item.i * self.grid_size)
        surface = (j , i, self.grid_size, self.grid_size)
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
                             (i * self.grid_size, j * self.grid_size, self.grid_size, self.grid_size), 0)

    def check_pygame_exit(self) -> bool:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return False
        return True
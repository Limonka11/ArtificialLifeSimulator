import numpy as np
from copy import deepcopy
from typing import List, Type

from entities import Entity, EntityTypes, Water

class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.entity_type = EntityTypes
        self.water_coordinates = []

        self.grid = np.zeros([self.height, self.width], dtype=object)
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i,j] = Entity((i, j), entity_type=self.entity_type.empty)

    # Return a cell from the map (grid)
    def get_cell(self, i: int, j: int) -> np.array:
        assert i < self.height and i >= 0
        assert j < self.width and j >= 0

        return self.grid[i,j]
        
    # Set a cell in the map (grid)
    def set_cell(self, i: int, j: int, entity: Type[Entity], **kwargs) -> np.array:
        assert i < self.height and i >= 0
        assert j < self.width and j >= 0

        self.grid[i,j] = entity((i, j), **kwargs)
        return self.grid[i,j]
    
    def copy(self):
        return deepcopy(self)
    
    def get_grid_as_numpy(self, entity_type: int = None) -> np.array:
        if entity_type:
            vectorized = np.vectorize(lambda obj: obj.entity_type == entity_type)
        else:
            vectorized = np.vectorize(lambda obj: obj.entity_type)
        return vectorized(self.grid)

    def get_entities(self, entity_type: int = None) -> List[Entity]:
        grid = self.get_grid_as_numpy()
        positions = np.where(grid == entity_type)
        positions = [(i, j) for i, j in zip(positions[0], positions[1])]
        entities = [self.grid[position] for position in positions]

        return entities

    def set_water(self):
        for w in range(self.width):
            for h in range(self.height):
                # Calculate the distance of the pixel from the center
                dist = np.sqrt((w - self.width/2)**2 + (h - self.height/2)**2)
                # Check if the pixel is inside the circle
                if dist <= 10:
                    # Set the color of the pixel
                    self.grid[w, h] = Water((w, h))
                    self.water_coordinates.append((w, h))

    def set_random(self, entity: Type[Entity], p: float, **kwargs) -> np.array:
        grid = self.get_grid_as_numpy()
        indices = np.where(grid == self.entity_type.empty)

        try:
            random_index = np.random.randint(0, len(indices[0]))
            i, j = indices[0][random_index], indices[1][random_index]
            if np.random.random() < p:
                self.grid[i, j] = entity((i, j), **kwargs)
                return self.grid[i, j]
            else:
                return None
        except ValueError:
            return None

    def update(self, i: int, j: int, entity_1: Type[Entity], k: int, l: int, entity_2: Type[Entity]):
        self.grid[i, j] = entity_1
        self.grid[k, l] = entity_2

    def get_surroundings(self, i: int, j: int, dist: int, grid: np.ndarray = None) -> np.ndarray:
        if grid is None:
            grid = self.grid

        # Get sections
        top = grid[:dist, :]
        bottom = grid[self.height - dist:, :]
        right = grid[:, self.width - dist:]
        left = grid[:, :dist]
        lower_left = grid[self.height - dist:, :dist]
        lower_right = grid[self.height - dist:, self.width - dist:]
        upper_left = grid[:dist, :dist]
        upper_right = grid[:dist, self.width - dist:]

        # Create top, middle and bottom sections
        full_top = np.concatenate((lower_right, bottom, lower_left), axis=1)
        full_middle = np.concatenate((right, grid, left), axis=1)
        full_bottom = np.concatenate((upper_right, top, upper_left), axis=1)

        # Apply fov based on dist
        surroundings = np.concatenate((full_top, full_middle, full_bottom), axis=0)
        surroundings = surroundings[i:i + (2 * dist) + 1, j:j + (2 * dist) + 1]

        return surroundings
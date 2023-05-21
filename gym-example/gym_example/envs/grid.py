import numpy as np
from typing import List, Type

from .entities import Entity, EntityTypes, Water, Tree

class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros([self.height, self.width], dtype=object)

        self.water_coordinates = []
        self.tree_coordinates = []

        for i in range(self.height):
            for j in range(self.width):
                self.grid[i,j] = Entity((i, j), entity_type = EntityTypes.empty)

    # Return a cell from the map (grid)
    def get_cell(self, i: int, j: int) -> np.array:
        return self.grid[i,j]

    def get_grid(self, entity_type: int = None) -> np.array:
        if entity_type:
            get_grid = np.vectorize(lambda obj: obj.entity_type == entity_type)
        else:
            get_grid = np.vectorize(lambda obj: obj.entity_type)
        return get_grid(self.grid)

    def get_entities(self, entity_type: int = None) -> List[Entity]:
        grid = self.get_grid()

        # Get positions of objects
        positions = np.where(grid == entity_type)
        positions = [(i, j) for i, j in zip(positions[0], positions[1])]
        
        # Get the objects from their positions
        entities = [self.grid[position] for position in positions]

        return entities
    
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

    def set_water(self):
        for w in range(self.width):
            for h in range(self.height):
                # Calculate the distance of the pixel from the center
                dist = np.sqrt((w - self.width/2)**2 + (h - self.height/2)**2)
                # Check if the pixel is inside the circle
                if dist <= 8:
                    # Set the color of the pixel
                    self.grid[w, h] = Water((w, h))
                    self.water_coordinates.append((w, h))

    def set_random(self, entity: Type[Entity], p: float, **kwargs) -> np.array:
        grid = self.get_grid()
        indices = np.where(grid == EntityTypes.empty)

        random_idx = np.random.randint(0, len(indices[0]))
        i, j = indices[0][random_idx], indices[1][random_idx]
        if np.random.random() < p:
            self.grid[i, j] = entity((i, j), **kwargs)
            if type(self.grid[i, j]) == Tree:
                self.tree_coordinates.append((i, j))
            return self.grid[i, j]
        else:
            return None
        
    # Set a cell in the map (grid)
    def set_cell(self, i: int, j: int, entity: Type[Entity], **kwargs) -> np.array:
        self.grid[i,j] = entity((i, j), **kwargs)
        return self.grid[i,j]
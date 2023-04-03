from typing import Collection
from enum import IntEnum
import pygame
import sys

sys.path.append("C:\\imperial\\MengProject\\Entities")

from Brain import Brain

class EntityTypes(IntEnum):
    empty = 0
    food = 1
    poison = 2
    agent = 3
    water = 4
    corpse = 5

class Actions(IntEnum):
    up = 0
    right = 1
    down = 2
    left = 3

    reproduce = 4

    attack_up = 5
    attack_right = 6
    attack_down = 7
    attack_left = 8

class Entity:
    # Represents a basic entity on the map

    def __init__(self, position: Collection[int], entity_type: int):
        assert len(position) == 2

        self.i, self.j = position
        self.new_i = None
        self.new_j = None
        self.entity_type = entity_type

class Empty(Entity):
    # Represent an empty space on the map

    def __init__(self, position: Collection[int]):
        super().__init__(position, EntityTypes.empty)
        self.nutrition = 0

class BasicFood(Entity):
    def __init__(self, coordinates: Collection[int], entity_type: int):
        super().__init__(coordinates, entity_type)
        self._nutrition = 0

    @property
    def nutrition(self) -> int:
        return self._nutrition

    @nutrition.setter
    def nutrition(self, val):
        self._nutrition = val

class Food(BasicFood):
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, EntityTypes.food)
        self.nutrition = 40

class Corpse(BasicFood):
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, EntityTypes.corpse)
        self.nutrition = 100

class Poison(BasicFood):
    def __init__(self, position):
        super().__init__(position, EntityTypes.poison)
        self.nutrition = -40

class Water(Entity):
    def __init__(self, position):
        super().__init__(position, EntityTypes.water)
        self.water = 100

class Agent(Entity):
    def __init__(self,
                position: Collection[int],
                brain: Brain = None,
                gene: int = None):
        super().__init__(position, EntityTypes.agent)

        self.health = 200
        self.max_health = 200
        self.age = 0
        self.max_age = 100000
        self.brain = brain
        self.reproduced = False
        self.gene = gene
        self.action = -1
        self.killed = 0
        self.dead = False
        self.hunger = 100
        self.thirst = 100
        self.max_thirst = 100
        self.max_hunger = 100
        self.birth_delay = 10
        
        # test feature
        self.has_eaten = 0
        self.has_reproduced = 0

        self.state = None
        self.state_prime = None
        self.reward = None
        self.done = False
        self.prob = None
        self.fitness = 0

    # Move the entity to the new location
    def move(self):
        self.i = self.new_i
        self.j = self.new_j

    # TODO: Change dynamics
    def execute_attack(self):
        self.health = min(200, self.health + 100)
        self.killed = 1

    # Decreas health if agent is attacked
    def is_attacked(self):
        self.health = 0

    # Update the new location
    def update_new_location(self, i: int, j: int):
        self.new_i = i
        self.new_j = j

    def update_rl_stats(self, reward: float, done: bool):
        self.fitness += reward
        self.reward = reward
        self.done = done

    def learn(self, n_epi, **kwargs):
        if self.age > 1:
            self.brain.learn(age=self.age,
                            dead=self.dead,
                            action=self.action,
                            state=self.state,
                            reward=self.reward,
                            state_prime=self.state_prime,
                            done=self.done,
                            n_epi=n_epi)

    def can_breed(self) -> bool:
        if not self.dead and self.age > 5 and self.hunger > 30 and self.birth_delay == 0:
            return True
        return False
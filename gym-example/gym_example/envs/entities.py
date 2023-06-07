from typing import Collection
from enum import IntEnum
import pygame
import sys
import numpy as np
import math

sys.path.append("C:\\imperial\\MengProject\\Entities")

from Brain import Brain

class EntityTypes(IntEnum):
    empty = 0
    food = 1
    poison = 2
    agent = 3
    water = 4
    corpse = 5
    pheromone = 6
    wolf = 7
    tree = 8

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
    def __init__(self, position: Collection[int], entity_type: int):
        assert len(position) == 2

        self.i, self.j = position
        self.new_i = None
        self.new_j = None
        self.entity_type = entity_type

class Empty(Entity):
    def __init__(self, position: Collection[int]):
        super().__init__(position, EntityTypes.empty)
        self.nutrition = 0

class Tree(Entity):
    def __init__(self, position: Collection[int]):
        super().__init__(position, EntityTypes.tree)
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

class Pheromone(BasicFood):
    def __init__(self, coordinates: Collection[int]):
        super().__init__(coordinates, EntityTypes.pheromone)
        self.lasting_time = 7

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
                gene: int = None,
                attack_damage: int = 200,
                armor: int = 0,
                agility = 0.05):
        super().__init__(position, EntityTypes.agent)

        self.id = ""
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
        self.libido = 0
        
        # test feature
        self.has_eaten = 0
        self.has_drunk = False
        self.has_reproduced = 0
        self.miss_reproduced = False
        self.miss_attacked = False
        self.inter_attacked = False

        # Evolutionary features
        self.attack_damage = attack_damage
        self.armor = armor
        self.agility = agility

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

    def execute_attack(self, agent_armor, agent_health, dodged):
        self.health = min(200, self.health + 100)

        if agent_armor + agent_health <= self.attack_damage and not dodged:
            self.killed = 1
            return True
        
        return False

    # Decreas health if agent is attacked
    def is_attacked(self, agent_attack_damage):
        damage_dealt = agent_attack_damage - self.armor

        dodged = True if np.random.random() < self.agility else False

        if damage_dealt > 0 and not dodged:
            self.health = max(0, self.health - damage_dealt)

        return dodged

    # Update the new location
    def update_new_location(self, i: int, j: int):
        self.new_i = i
        self.new_j = j

    def update_rl_stats(self, reward: float, done: bool):
        self.fitness += reward
        self.reward = reward
        self.done = done

    # Currently not used!
    # This was used when Neuroevolution was tested
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
        self.libido = self.get_agent_libido()
        if (not self.dead) and self.age > 40 and self.hunger > 30 and self.birth_delay == 0 and self.libido > 5.:
            return True
        return False
    
    def get_agent_libido(self):
        libido = 0
        real_age = self.age // 4

        if real_age >= 25 and real_age < 100:
            libido = 10 * math.exp((-(real_age - 20) / 40)) + 2 * math.exp((-(real_age - 20) / 20)) 
        elif real_age >= 0 and real_age < 25:
            libido = (real_age ** 2) / 60

        libido /= (self.birth_delay + 1)
        return libido
    
class Wolf(Agent):
    def __init__(self,
            position: Collection[int],
            brain: Brain = None,
            gene: int = None,
            attack_damage = 200,
            armor = 0,
            agility = 0.05):
        super().__init__(position, brain, gene, attack_damage, armor, agility)

        self.hunger = 200
        self.thirst = 200
        self.max_thirst = 200
        self.max_hunger = 200
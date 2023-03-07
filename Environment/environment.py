import collections
import random
import copy
import numpy as np
from typing import List, Type, Union, Collection, Tuple
import torch
import sys

sys.path.append("C:\\imperial\\MengProject\\Entities")
from DQNBrain import DQNBrain
from A2CBrain import DDQNBrain
from grid import Grid
from entities import Agent, Entity, Food, Poison, Empty, EntityTypes, Actions
from Brain import Brain
sys.path.append("C:\\imperial\\MengProject\\Renderer")
from SimRenderer import SimRenderer

class Environment:

    def __init__(self,
                width: int = 30,
                height: int = 30,
                brains: List[Brain] = None,
                grid_size: int = 16,
                max_agents: int = 20,
                update_interval: int = 500,
                print_results: bool = True,
                training: bool = True,
                save: bool = False):

        self.width = width
        self.height = height
        self.grid = None

        self.actions = Actions
        self.entities = EntityTypes

        self.agents = []
        self.brains = brains

        self.max_agents = max_agents
        self.max_gene = len(brains)

        self.SimRenderer = SimRenderer(self.width, self.height, grid_size)

    def init_environment(self):
        self.grid = Grid(self.width, self.height)
        self.agents = [self.add_agent(random_loc=True, brain=self.brains[i], gene=i) for i in range(self.max_gene)]
        self.init_water()
        self.init_consumables(Food, probability = 0.1)
        self.init_consumables(Poison, probability = 0.05)
        self.get_observations()
        self.update_agent_state()

    def update_step(self):
        self.act()
        self.update_dead_agents()
        self.get_rewards()
        self.add_food()
        self.get_observations()

    def update_environment(self):
        self.agents = self.grid.get_entities(self.entities.agent)
        self.reproduce()
        self.generate_agent()
        self.remove_dead_agents()
        self.get_observations()
        self.update_agent_state()

    def act(self):
        for agent in self.agents:
            agent.age = min(agent.max_age, agent.age + 1)
            #agent.reset_killed()

        #self.attack()
        self.choose_action()
        self.execute_movement()

    def get_rewards(self):
        for agent in self.agents:
            done = False
            nr_kin_alive = max(0, sum([1 for other_agent in self.agents if not other_agent.dead and
                                       agent.gene == other_agent.gene]) - 1)
            alive_agents = sum([1 for other_agent in self.agents if not other_agent.dead])
            if agent.dead:
                reward = (-1 * alive_agents) + nr_kin_alive
                done = True
                print("not here not")

            elif alive_agents == 1:
                reward = 0
                print("not here")
            else:
                # Percentage of population with the same kin + cur health/max health
                #reward = (nr_kin_alive / alive_agents) + (agent.health / agent.max_health)
                print("here")
                reward = (agent.health / agent.max_health) + (agent.hunger / agent.max_hunger) + (agent.thirst / agent.max_thirst)
            if agent.killed and self.incentivize_killing:
                reward += 0.2
            agent.update_rl_stats(reward, done)
            
    def get_observations(self):
        self.agents = self.grid.get_entities(self.entities.agent)
        
        vectorize = np.vectorize(lambda obj: self.get_food(obj))
        food_grid = vectorize(self.grid.grid)

        vectorize = np.vectorize(lambda obj: self.get_water(obj))
        water_grid = vectorize(self.grid.grid)

        vectorize = np.vectorize(
            lambda obj: obj.health / obj.max_health if obj.entity_type == self.entities.agent else -1)
        health_grid = vectorize(self.grid.grid)

        for agent in self.agents:
            reproduced = agent.birth_delay == 10

            food_obs = list(self.grid.get_surroundings(agent.i, agent.j, 16, food_grid).flatten())
            water_obs = list(self.grid.get_surroundings(agent.i, agent.j, 16, water_grid).flatten())
            health_obs = list(self.grid.get_surroundings(agent.i, agent.j, 16, health_grid).flatten())
            # Combine all observations
            observation = np.array(food_obs +
                                   health_obs +
                                   water_obs +
                                   [agent.health / agent.max_health] +
                                   [agent.thirst / agent.max_thirst] +
                                   [agent.hunger / agent.max_hunger] +
                                   [reproduced])
            # Update their states
            if agent.age == 0:
                agent.state = observation
            agent.state_prime = observation
        
    def get_food(self, obj: Entity or Agent) -> float:
        if obj.entity_type == self.entities.food:
            return .5
        elif obj.entity_type == self.entities.poison:
            return -1.
        elif obj.entity_type == self.entities.agent:
            if obj.health < 0:
                return 1.
            else:
                return 0.
        else:
            return 0.

    def get_water(self, obj: Entity or Agent) -> float:
        if obj.entity_type == self.entities.water:
            return 1.
        else:
            return 0.

    def get_gene(self, obj: Entity or Agent) -> int:
        if obj.entity_type == self.entities.agent:
            if not obj.dead:
                return -2
            else:
                return obj.gene
        else:
            return -2

    def choose_action(self):
        for agent in self.agents:
            if not agent.dead and agent.action <= 3:
                if agent.action == Actions.up:
                    if agent.i != 0:
                        agent.update_new_location(agent.i - 1, agent.j)
                    else:
                        agent.update_new_location(self.height - 1, agent.j)
            
                if agent.action == Actions.down:
                    if agent.i != self.height - 1:
                        agent.update_new_location(agent.i + 1, agent.j)
                    else:
                        agent.update_new_location(0, agent.j)
                if agent.action == Actions.left:
                    if agent.j != 0:
                        agent.update_new_location(agent.i, agent.j - 1)
                    else:
                        agent.update_new_location(agent.i, self.width - 1)
                if agent.action == Actions.right:
                    if agent.j != self.width - 1:
                        agent.update_new_location(agent.i, agent.j + 1)
                    else:
                        agent.update_new_location(agent.i, 0)
            else:
                agent.update_new_location(agent.i, agent.j)

    def execute_movement(self):
        impossible_coordinates = True
        while impossible_coordinates:

            impossible_coordinates = self.get_impossible_coordinates()
            for agent in self.agents:
                if (agent.new_i, agent.new_j) in impossible_coordinates:
                    agent.update_new_location(agent.i, agent.j)
                
                if (agent.new_i, agent.new_j) in self.grid.water_coordinates:
                    agent.update_new_location(agent.i, agent.j)
        
        for agent in self.agents:
            if agent.action <= 3:
                self.eat(agent)
                self.drink(agent)
                self.update_agent_position(agent)

    def drink(self, agent: Agent):
        if agent.i + 1 < self.grid.width and agent.i - 1 > -1 and agent.j + 1 <  self.grid.height and agent.j - 1 > -1 and self.entities.water in [
            self.grid.grid[agent.i-1, agent.j].entity_type,
            self.grid.grid[agent.i, agent.j-1].entity_type,
            self.grid.grid[agent.i+1, agent.j].entity_type,
            self.grid.grid[agent.i, agent.j+1].entity_type]:

            agent.thirst = 100
    
    def eat(self, agent: Agent):
        if self.grid.grid[agent.new_i, agent.new_j].entity_type == self.entities.food:
            agent.hunger = min(100, agent.hunger + 40)
            agent.hunger = max(0, agent.hunger)

        elif self.grid.grid[agent.new_i, agent.new_j].entity_type == self.entities.poison:
            agent.hunger = min(100, agent.hunger + 20)
            agent.hunger = max(0, agent.hunger)
            agent.health = min(200, agent.health - 30)

    def init_consumables(self,
                        entity: Union[Type[Food], Type[Poison]],
                        probability: float = 0.1):
            
        entity_type = entity([-1, -1]).entity_type
        for _ in range(self.width * self.height):
            if np.random.random() < probability:
                self.grid.set_random(entity, p=1)

    def init_water(self):
        self.grid.set_water()

    def add_food(self):
        if len(np.where(self.grid.get_grid_as_numpy() == self.entities.food)[0]) <= ((self.width * self.height) / 10):
            for _ in range(3):
                self.grid.set_random(Food, p=0.2)

        if len(np.where(self.grid.get_grid_as_numpy() == self.entities.poison)[0]) <= ((self.width * self.height) / 20):
            for _ in range(3):
                self.grid.set_random(Poison, p=0.2)
    
    def update_agent_position(self, agent: Agent):
        self.grid.grid[agent.i, agent.j] = Empty((agent.i, agent.j))
        self.grid.grid[agent.new_i, agent.new_j] = agent
        agent.move()

    def update_agent_state(self):
        for agent in self.agents:
            agent.state = agent.state_prime

            # Take damage if hungry
            if agent.hunger == 0:
                agent.health -= 5

            if agent.thirst == 0:
                agent.health -= 5

            # Heal if not hungry and not thirsty
            if agent.hunger >= 50 and agent.thirst >= 50:
                agent.health += 5
            
            # Heal if not hungry only
            elif agent.hunger >= 50:
                agent.health += 2

            # Heal if not thirsty only
            elif agent.thirst >= 50:
                agent.health += 2


            # The agent must get hungry and thirsty after so much running! Phew...
            agent.hunger = max(0, agent.hunger - 5)
            agent.thirst = max(0, agent.thirst - 5)

            # Reduce the birth delay
            agent.birth_delay = max(0, agent.birth_delay - 1)
    
    def update_dead_agents(self):
        for agent in self.agents:
            if agent.health <= 0 or agent.age == agent.max_age:
                agent.dead = True

    def remove_dead_agents(self):
        for agent in self.agents:
            if agent.dead:
                self.grid.grid[agent.i, agent.j] = Food((agent.i, agent.j))
    def render(self, fps: int = 10) -> bool:
        return self.SimRenderer.render(self.agents, self.grid, fps=fps)
    def add_agent(self,
               coordinates: Collection[int] = None,
               brain: Brain = None,
               gene: int = None,
               random_loc: bool = False,
               p: float = 1.) -> Type[Entity] or None:
        if random_loc:
            return self.grid.set_random(Agent, p=p, brain=brain, gene=gene)
        else:
            return self.grid.set_cell(coordinates[0], coordinates[1], Agent, brain=brain, gene=gene)

    def get_impossible_coordinates(self):
        new_coordinates = [(agent.new_i, agent.new_j) for agent in self.agents]
        if new_coordinates:
            unq, count = np.unique(new_coordinates, axis=0, return_counts=True)
            impossible_coordinates = [(coordinate[0], coordinate[1]) for coordinate in unq[count > 1]]
            return impossible_coordinates
        else:
            return []

    def get_empty_cells_surroundings(self, agent: Agent) -> List[Union[int, int]] or List[None]:
        observation = self.grid.get_surroundings(agent.i, agent.j, 3)

        loc = []
        for i in range(len(observation)):
            for j in range(len(observation[1])):
                if observation[i][j].entity_type == 0:
                    loc.append((i, j))

        #loc = np.where(type(observation[:,:]) == Empty)
        coordinates = []

        for i_local, j_local in loc:
            diff_x = i_local - 3
            diff_y = j_local - 3

            # Get coordinates if within normal range
            global_x = agent.i + diff_x
            global_y = agent.j + diff_y

            # Get coordinates if through wall (left vs right)
            if global_y < 0:
                global_y = self.width + global_y
            elif global_y >= self.width:
                global_y = global_y - self.width

            # Get coordinates if through wall (up vs down)
            if global_x < 0:
                global_x = self.height + global_x
            elif global_x >= self.height:
                global_x = global_x - self.height

            coordinates.append([global_x, global_y])

        return coordinates

    def get_adjacent_agent(self, agent1: Agent) -> Agent:
        agents = [(agent.i, agent.j, agent) for agent in self.agents]

        for i in range(len(agents)):
            x, y, agent2 = agents[i]
            if agent1 != agent2 and abs(x - agent1.i) <= 3 and abs(y - agent1.j) <= 3:
                return agent2


    def reproduce(self):
        for agent in self.agents:
            adjacent_agent = self.get_adjacent_agent(agent)
            if agent.can_reproduce() and \
               len(self.agents) <= self.max_agents and \
               adjacent_agent and \
               type(agent.brain) is type(adjacent_agent.brain):
                new_brain = self.crossover(agent, adjacent_agent)

                agent.birth_delay = 10
                adjacent_agent.birth_delay = 10

                # Add offspring close to parent
                coordinates = self.get_empty_cells_surroundings(agent)
                if coordinates:
                    self.add_agent(coordinates=coordinates[random.randint(0, len(coordinates) - 1)],
                                    brain=new_brain, gene=agent.gene)
                else:
                    self.add_agent(random_loc=True, brain=new_brain, gene=agent.gene)

    # Add new agents if the population is too small. The brain of the new agent will be copied from 
    # the current oldest agent
    def generate_agent(self):
        new_agent = None
        for agent in self.agents:
            if new_agent == None or new_agent.age < agent.age:
                new_agent = agent

        if len(self.agents) < 10:
            self.add_agent(random_loc=True, brain=new_agent.brain, gene=new_agent.gene)

    def crossover_weights(self, weights1, weights2):
        new_weights = collections.OrderedDict()

        for key in weights1.keys():
            # Randomly choose a crossover point
            crossover_point = np.random.choice(range(len(weights1[key])))
            # Create the new weight tensor
            new_weight = torch.cat((weights1[key][:crossover_point], weights2[key][crossover_point:]), dim=0)
            new_weights[key] = new_weight
            
        return new_weights
    
    def crossover(self, agent1, agent2):
        print(isinstance(agent1.brain, DDQNBrain))
        child_brain = DQNBrain() if isinstance(agent1.brain, DQNBrain) else DDQNBrain()
        weights1 = agent1.brain.agent.state_dict()
        weights2 = agent2.brain.agent.state_dict()
        # Apply crossover on the weights
        new_weights = self.crossover_weights(weights1, weights2)
        child_brain.agent.load_state_dict(new_weights)
        child_brain.target.load_state_dict(new_weights)
        # Mutate the genes of the child brain
        child_brain.mutate()
        return child_brain
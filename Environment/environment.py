import collections
import random
import copy
import numpy as np
from typing import List, Type, Union, Collection, Tuple
import torch
import sys
import matplotlib.pyplot as plt
import os

sys.path.append("C:\\imperial\\MengProject\\Entities")
from DQNBrain import DQNBrain
from A2CBrain import DDQNBrain
from grid import Grid
from entities import Agent, Entity, Food, Poison, Empty, Corpse, EntityTypes, Actions
from Brain import Brain, RandBrain
sys.path.append("C:\\imperial\\MengProject\\Renderer")
from SimRenderer import SimRenderer
import time

np.set_printoptions(threshold=sys.maxsize)

class Environment:

    def __init__(self,
                width: int = 30,
                height: int = 30,
                brains: List[Brain] = None,
                square_size: int = 32,
                max_agents: int = 20,
                training: bool = False):

        self.width = width
        self.height = height
        self.grid = None

        self.SimRenderer = SimRenderer(self.width, self.height, square_size)
        self.entities = EntityTypes
        self.agents = []
        self.brains = brains
        self.actions = Actions

        self.max_agents = max_agents
        self.max_gene = len(brains)
        self.oldest_age = 0

        # Metrics
        self.dead_agents = []
        self.dead_scores = []
        self.lives = []
        self.gens = [[],[],[]]
        self.max_avg_age = 0
        self.max_old_age = 0
    
    def init_environment(self):
        self.grid = Grid(self.width, self.height)
        self.agents = [self.add_agent(random_loc=True, brain=self.brains[i], gene=i+1) for i in range(self.max_gene)]
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
        #self.breed()
        self.remove_dead_agents()
        self.generate_agent()
        self.get_observations()
        self.update_agent_state()

    def act(self):
        for agent in self.agents:
            agent.age = min(agent.max_age, agent.age + 1)
            #agent.reset_killed()

        self.attack()
        self.choose_action()
        self.move()

    def get_rewards(self):
        for agent in self.agents:
            done = False
            nr_kin_alive = max(0, sum([1 for other_agent in self.agents if not other_agent.dead and
                                       agent.gene == other_agent.gene]) - 1)
            alive_agents = sum([1 for other_agent in self.agents if not other_agent.dead])
            if agent.dead:
                reward = (-1 * alive_agents) + nr_kin_alive
                done = True

            elif alive_agents == 1:
                reward = 0
            else:
                # Percentage of population with the same kin + cur health/max health
                reward = (nr_kin_alive / alive_agents)
                #reward = (agent.health / agent.max_health) + (agent.hunger / agent.max_hunger) + (agent.thirst / agent.max_thirst)
                #reward = agent.has_eaten
                #print(reward)
            if agent.killed:
                reward += 0.2

            agent.update_rl_stats(reward, done)
            
    def get_observations(self):
        self.agents = self.grid.get_entities(self.entities.agent)
        
        # Get separate grid for separate entities
        get_gene = np.vectorize(lambda item: self.get_gene(item))
        gene_grid = get_gene(self.grid.grid)

        get_food = np.vectorize(lambda item: self.get_food(item))
        food_grid = get_food(self.grid.grid)

        get_water = np.vectorize(lambda item: self.get_water(item))
        water_grid = get_water(self.grid.grid)

        get_health = np.vectorize(
            lambda obj: obj.health / obj.max_health if obj.entity_type == self.entities.agent else -1)
        health_grid = get_health(self.grid.grid)

        for agent in self.agents:
            reproduced = agent.birth_delay == 50

            gene_observed = self.grid.get_surroundings(agent.i, agent.j, 3, gene_grid)
            gene_observed[(gene_observed != 0) & (gene_observed != agent.gene)] = -1
            gene_observed[gene_observed == agent.gene] = 1
            gene_observed = list(gene_observed.flatten())
            food_observed = list(self.grid.get_surroundings(agent.i, agent.j, 3, food_grid).flatten())
            water_observed = list(self.grid.get_surroundings(agent.i, agent.j, 3, water_grid).flatten())
            health_observed = list(self.grid.get_surroundings(agent.i, agent.j, 3, health_grid).flatten())

            observation = np.array(gene_observed +
                                   food_observed +
                                   health_observed +
                                   water_observed +
                                   [agent.health / agent.max_health] +
                                   [agent.thirst / agent.max_thirst] +
                                   [agent.hunger / agent.max_hunger] +
                                   [reproduced])
            
            # Update their states
            agent.state_prime = observation

            if agent.age == 0:
                agent.state = observation

    def get_gene(self, obj: Entity or Agent) -> int:
        if obj.entity_type == self.entities.agent and not obj.dead:
            return obj.gene
        else:
            return 0

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

    def attack(self):
        """ Attack and decrease health if target is hit
        The agent is killed if attacked.
        Whether an agent killed kin or not is tracked.
        """
        for agent in self.agents:
            target = Empty((-1, -1))

            if not agent.dead:

                # Attack up
                if agent.action == self.actions.attack_up:
                    #print("ATTACK_UP")
                    if agent.i == 0:
                        target = self.grid.get_cell(self.height - 1, agent.j)
                    else:
                        target = self.grid.get_cell(agent.i - 1, agent.j)

                # Attack right
                elif agent.action == self.actions.attack_right:
                    #print("ATTACK_RIGHT")
                    if agent.j == (self.width - 1):
                        target = self.grid.get_cell(agent.i, 0)
                    else:
                        target = self.grid.get_cell(agent.i, agent.j + 1)

                # Attack down
                elif agent.action == self.actions.attack_down:
                    #print("ATTACK_DOWN")
                    if agent.i == (self.height - 1):
                        target = self.grid.get_cell(0, agent.j)
                    else:
                        target = self.grid.get_cell(agent.i + 1, agent.j)

                # Attack left
                elif agent.action == self.actions.attack_left:
                    #print("ATTACK_LEFT")
                    if agent.j == 0:
                        target = self.grid.get_cell(agent.i, self.width - 1)
                    else:
                        target = self.grid.get_cell(agent.i, agent.j - 1)

                # Execute attack
                if target.entity_type == self.entities.agent:
                    target.is_attacked()
                    agent.execute_attack()

                    if target.gene == agent.gene:
                        agent.inter_killed = 1
                    else:
                        agent.intra_killed = 1

    def choose_action(self):
        for agent in self.agents:
            if not agent.dead and agent.action <= 4:
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
                if agent.action == Actions.reproduce:
                    agent.update_new_location(agent.i, agent.j)
                    agent.has_reproduced = -10
                    adjacent_agent = self.get_adjacent_agent(agent)
                    if agent.can_breed() and \
                       len(self.agents) <= self.max_agents and \
                       adjacent_agent and \
                       type(agent.brain) is type(adjacent_agent.brain):
                        #print("REPRODUCE REPRODUCE REPRODUCE")
                        new_brain = self.crossover(agent, adjacent_agent)

                        agent.birth_delay = 50
                        adjacent_agent.birth_delay = 50

                        # Add offspring close to parent
                        coordinates = self.get_empty_cells_surroundings(agent)

                        # Update the reward value
                        agent.has_reproduced = 10

                        if coordinates:
                            self.add_agent(coordinates=coordinates[random.randint(0, len(coordinates) - 1)],
                                            brain=new_brain, gene=agent.gene)
                        else:
                            self.add_agent(random_loc=True, brain=new_brain, gene=agent.gene)
                    
            else:
                agent.update_new_location(agent.i, agent.j)

    def move(self):
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
            #agent.has_eaten = 10

        elif self.grid.grid[agent.new_i, agent.new_j].entity_type == self.entities.poison:
            agent.hunger = min(100, agent.hunger + 10)
            agent.hunger = max(0, agent.hunger)
            agent.health = min(200, agent.health - 10)
            #agent.has_eaten = -10

        #else:
            #agent.has_eaten = -1

    def init_consumables(self,
                        entity: Union[Type[Food], Type[Poison]],
                        probability: float):
            
        entity_type = entity([-1, -1]).entity_type
        for _ in range(self.width * self.height):
            if np.random.random() < probability:
                self.grid.set_random(entity, p=1)

    def init_water(self):
        self.grid.set_water()

    def add_food(self):
        if len(np.where(self.grid.get_grid() == self.entities.food)[0]) <= ((self.width * self.height) / 10):
            for _ in range(3):
                self.grid.set_random(Food, p=0.2)

        if len(np.where(self.grid.get_grid() == self.entities.poison)[0]) <= ((self.width * self.height) / 20):
            for _ in range(3):
                self.grid.set_random(Poison, p=0.2)
    
    def update_agent_position(self, agent: Agent):
        self.grid.grid[agent.i, agent.j] = Empty((agent.i, agent.j))
        self.grid.grid[agent.new_i, agent.new_j] = agent
        agent.move()

    def update_agent_state(self):
        for agent in self.agents:
            agent.state = agent.state_prime

            # Update oldest age
            if agent.age > self.oldest_age:
                self.oldest_age = agent.age
                #print(self.oldest_age)

            # Take damage if hungry
            if agent.hunger == 0:
                agent.health -= 5

            # if agent.thirst == 0:
            #     agent.health -= 5

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
            agent.hunger = max(0, agent.hunger - 2)
            agent.thirst = max(0, agent.thirst - 2)

            # Reduce the birth delay
            agent.birth_delay = max(0, agent.birth_delay - 1)
    
    def update_dead_agents(self):
        for agent in self.agents:
            if agent.health <= 0:# or agent.age == agent.max_age:
                agent.dead = True

    def remove_dead_agents(self):
        for agent in self.agents:
            if agent.dead:
                # Update statistics
                self.dead_agents.append(agent)
                self.dead_scores.append(agent.age)

                # TODO: Add more nutritional food (corpse)
                self.grid.grid[agent.i, agent.j] = Corpse((agent.i, agent.j))
    
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
            if agent1 != agent2 and abs(x - agent1.i) <= 2 and abs(y - agent1.j) <= 2:
                return agent2


    # Used for breeding without the choice of agents
    def breed(self):
        for agent in self.agents:
            adjacent_agent = self.get_adjacent_agent(agent)
            if agent.can_breed() and \
               len(self.agents) <= self.max_agents and \
               adjacent_agent and \
               type(agent.brain) is type(adjacent_agent.brain):
                new_brain = self.crossover(agent, adjacent_agent)

                agent.birth_delay = 50
                adjacent_agent.birth_delay = 50

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

        if len(self.agents) == 1:
            # Record statistics for the current generation
            max_age = max(self.dead_scores)
            min_age = min(self.dead_scores)
            avg_age = np.average(self.dead_scores)

            # Check whether the model is worth saving
            if len(self.gens[0]) >= 5 and \
               self.max_old_age < max_age and \
               self.max_avg_age < avg_age:
                
                big_brain = self.dead_scores.index(max_age)
                self.save_brains(self.dead_agents[big_brain])

                self.max_old_age = max_age
                self.max_avg_age = avg_age

            self.gens[0].append(max_age)
            self.gens[1].append(min_age)
            self.gens[2].append(avg_age)

            # Sort the agents based on their scores, in descending order
            #print(self.dead_agents)
            sorted_agents = sorted(self.dead_agents, key=lambda x: x.age, reverse=True)

            best_agents = [sorted_agents[0]]
    
            # Compute the total score of all agents
            total_score = sum([agent.age for agent in best_agents])
    
            # Compute the selection probabilities for each agent, based on their scores
            selection_probs = [agent.age / total_score for agent in best_agents]

            for _ in range(self.max_agents//2):
                if not isinstance(self.agents[0].brain, RandBrain):
                    # Use a weighted random choice to select an agent, with higher chances for agents with higher scores
                    selected_agent_idx = np.random.choice(len(best_agents), p=selection_probs)
                    selected_agent = best_agents[selected_agent_idx]

                    old_brain_weights = selected_agent.brain.agent.state_dict()
                    for name, param in old_brain_weights.items():
                        if np.random.random() < 0.02: # mutation_rate
                            noise = torch.randn(param.shape) * 0.01 # mutation_std = 0.01
                            old_brain_weights[name] += noise
                    
                    new_brain = DQNBrain() if isinstance(selected_agent.brain, DQNBrain) else DDQNBrain()
                    new_brain.agent.load_state_dict(old_brain_weights)
                    new_brain.target.load_state_dict(old_brain_weights)

                    self.add_agent(random_loc=True, brain=new_brain, gene=selected_agent.gene)
                else:
                    self.add_agent(random_loc=True, brain=RandBrain(), gene=selected_agent.gene)
            
            # Reset metrics
            self.dead_agents = []
            self.dead_scores = []
            self.lives = []

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
        if not isinstance(agent1.brain, RandBrain) and \
            not isinstance(agent2.brain, RandBrain):
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
        else:
            return RandBrain()
    
    def show_stats(self):
        gens = range(len(self.gens[0]))
        max_ages = self.gens[0]
        min_ages = self.gens[1]
        avg_ages = self.gens[2]

        plt.plot(gens, max_ages)
        plt.plot(gens, min_ages)
        plt.plot(gens, avg_ages)
        plt.legend(['Max Age', 'Min Age', 'Avg Age'], loc='upper left')
        plt.show()

    def save_brains(self, agent: Agent):
        time_str = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(time_str, exist_ok = True) 
        torch.save({'model_state_dict': agent.brain.target.state_dict(),
                    'optimizer_state_dict': agent.brain.optimizer.state_dict()}, 
	                os.path.join(time_str,'model.pth'))

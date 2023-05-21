import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from typing import List, Type, Union, Collection, Tuple
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import matplotlib.pyplot as plt
import collections
import random
import torch
import os
import numpy as np
import pygame

from .grid import Grid
from .entities import Agent, Entity, Food, Poison, Empty, Corpse, Pheromone, Wolf, EntityTypes, Actions, Tree
from .Renderer.SimRenderer import SimRenderer
from Brain import Brain, RandBrain
from .Entities.DQNBrain import DQNBrain
from .Entities.A2CBrain import DDQNBrain

class Survival_Map(MultiAgentEnv):
    metadata_env = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: dict = {}):
        self.size = config.get("size", 50)
        self.square_size = 15
        self.grid = Grid(self.size, self.size)
        self.SimRenderer = SimRenderer(self.size, self.size, self.square_size)
        self.window_size = 512

        self.agents = []
        self.brains = config.get("brains", [DDQNBrain()])
        self.num_brains = len(self.brains)
        self.max_agents = config.get("max_agents", 20)

        # State of each agent
        self.state = {}
        self.agent_count = 0

        # Metrics
        self.dead_agents = []
        self.dead_scores = []
        self.lives = []
        self.gens = [[],[],[]]
        self.max_avg_age = 0
        self.max_old_age = 0

        # Change those with Gym Spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-6, high=50, shape=(300,), dtype=np.float32)

        render_mode = config.get("render_mode", "rgb_array")
        assert render_mode is None or render_mode in self.metadata_env["render_modes"]
        self.render_mode = config.get("render_mode")

        self.window = None
        self.clock = None

        self.reset()

    def reset(self, *, seed=None, options=None) -> tuple:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        self.agents = []
        self.state = {}
        infos = {}

        # Metrics
        self.dead_agents = []
        self.dead_scores = []
        self.lives = []
        self.gens = [[],[],[]]
        self.max_avg_age = 0
        self.max_old_age = 0

        # Reset Grid
        self.grid = Grid(self.size, self.size)

        # Reset water
        self._init_water()
        
        # Reset Consumables
        self._init_consumables(Food, probability = 0.1)
        self._init_consumables(Poison, probability = 0.05)
        self._init_trees(Tree, probability=0.02)

        # Reset agents
        self.agents = []
        for idx in range(self.num_brains):
            # Create a new agent and add in the the agent list
            cur_agent = self._add_agent(random_loc=True,
                                        brain=self.brains[idx],
                                        gene=(self.agent_count+1) % 2 + 10,
                                        type=Agent if (self.agent_count+1) % 3 != 0 else Wolf)
            cur_agent.id = self.agent_key(cur_agent.gene, self.agent_count, type(cur_agent))
            self.agents.append(cur_agent)
            self.state[cur_agent.id] = self._get_agent_obs(idx)
            infos[cur_agent.id] = {}

        return self.state, infos
    
    def step(self, action_dict):
        self.timesteps += 1

        # We get the length before starting the loop because
        # if the agents reproduce the list will grow
        agents_len = len(self.agents)
        for idx in range(agents_len):
            self.agents[idx].age = min(self.agents[idx].max_age, self.agents[idx].age + 1)

            target = Empty((-1, -1))

            if not self.agents[idx].dead and action_dict[self.agents[idx].id] <= 4:
                self.agents[idx].action = action_dict[self.agents[idx].id]
                if self.agents[idx].action == Actions.up:
                    if self.agents[idx].i != 0:
                        self.agents[idx].update_new_location(self.agents[idx].i - 1, self.agents[idx].j)
                    else:
                        self.agents[idx].update_new_location(self.size - 1, self.agents[idx].j)
            
                if self.agents[idx].action == Actions.down:
                    if self.agents[idx].i != self.size - 1:
                        self.agents[idx].update_new_location(self.agents[idx].i + 1, self.agents[idx].j)
                    else:
                        self.agents[idx].update_new_location(0, self.agents[idx].j)

                if self.agents[idx].action == Actions.left:
                    if self.agents[idx].j != 0:
                        self.agents[idx].update_new_location(self.agents[idx].i, self.agents[idx].j - 1)
                    else:
                        self.agents[idx].update_new_location(self.agents[idx].i, self.size - 1)

                if self.agents[idx].action == Actions.right:
                    if self.agents[idx].j != self.size - 1:
                        self.agents[idx].update_new_location(self.agents[idx].i, self.agents[idx].j + 1)
                    else:
                        self.agents[idx].update_new_location(self.agents[idx].i, 0)

                if self.agents[idx].action == Actions.reproduce:
                    self.agents[idx].update_new_location(self.agents[idx].i, self.agents[idx].j)
                    adjacent_agent = self._get_adjacent_agent(self.agents[idx])

                    if self.agents[idx].can_breed() and \
                       len(self.agents) <= self.max_agents and \
                       adjacent_agent and \
                       type(self.agents[idx].brain) is type(adjacent_agent.brain) and \
                       type(self.agents[idx]) is type(adjacent_agent):
                        #print(type(self.agents[idx]), " just reprocuded")
                        
                        #new_brain = self._crossover(agent, adjacent_agent)
                        new_brain = self.agents[idx].brain

                        # Set birth delay
                        self.agents[idx].birth_delay = 30
                        adjacent_agent.birth_delay = 30

                        # Add offspring close to parent
                        coordinates = self._get_empty_cells_surroundings(self.agents[idx])

                        # Update the reward value
                        self.agents[idx].has_reproduced = 10

                        if coordinates:
                            new_agent = self._add_agent(coordinates=coordinates[random.randint(0, len(coordinates) - 1)],
                                            brain=new_brain, gene=self.agents[idx].gene, type=type(self.agents[idx]))
                        else:
                            new_agent = self._add_agent(random_loc=True, brain=new_brain, gene=self.agents[idx].gene,
                                                        type=type(self.agents[idx]))

                        # Required because sometimes an id that has already existed is generated
                        new_agent.id = self.agent_key(self.agents[idx].gene, self.agent_count, type=type(self.agents[idx]))
                        
                        # Need to add the next location
                        new_agent.update_new_location(self.agents[idx].i, self.agents[idx].j)
                        #print("ADDED: ", new_agent.id)
                        self.agents.append(new_agent)
                        #print("DUCK:", [agent.id for agent in self.agents])
                    else:
                        self.agents[idx].miss_reproduced = True
                    
            elif not self.agents[idx].dead:
                self.agents[idx].update_new_location(self.agents[idx].i, self.agents[idx].j)
                self.agents[idx].action = action_dict[self.agents[idx].id]

                # Attack up
                if self.agents[idx].action == Actions.attack_up:
                    #print("ATTACK_UP")
                    if self.agents[idx].i == 0:
                        target = self.grid.get_cell(self.size - 1, self.agents[idx].j)
                    else:
                        target = self.grid.get_cell(self.agents[idx].i - 1, self.agents[idx].j)

                # Attack right
                elif self.agents[idx].action == Actions.attack_right:
                    #print("ATTACK_RIGHT")
                    if self.agents[idx].j == (self.size - 1):
                        target = self.grid.get_cell(self.agents[idx].i, 0)
                    else:
                        target = self.grid.get_cell(self.agents[idx].i, self.agents[idx].j + 1)

                # Attack down
                elif self.agents[idx].action == Actions.attack_down:
                    #print("ATTACK_DOWN")
                    if self.agents[idx].i == (self.size - 1):
                        target = self.grid.get_cell(0, self.agents[idx].j)
                    else:
                        target = self.grid.get_cell(self.agents[idx].i + 1, self.agents[idx].j)

                # Attack left
                elif self.agents[idx].action == Actions.attack_left:
                    #print("ATTACK_LEFT")
                    if self.agents[idx].j == 0:
                        target = self.grid.get_cell(self.agents[idx].i, self.size - 1)
                    else:
                        target = self.grid.get_cell(self.agents[idx].i, self.agents[idx].j - 1)

                # Execute attack
                if target.entity_type == EntityTypes.agent:
                    target.is_attacked()
                    self.agents[idx].execute_attack()
                    if type(self.agents[idx]) == Agent:
                        if target.gene == self.agents[idx].gene:
                            self.agents[idx].inter_killed = True
                        else:
                            self.agents[idx].inter_killed = False

                    elif type(self.agents[idx]) == Wolf:
                        self.agents[idx].has_eaten = 10

                if target.entity_type == EntityTypes.wolf:
                    if type(self.agents[idx]) == Agent:
                        self.agents[idx].dead = True

                    elif type(self.agents[idx]) == Wolf:
                        target.is_attacked()
                        self.agents[idx].execute_attack()
                        
                        if target.gene == self.agents[idx].gene:
                            self.agents[idx].inter_killed = True
                        else:
                            self.agents[idx].inter_killed = False
                else:
                    self.agents[idx].miss_attacked = True
            
            else:
                self.agents[idx].update_new_location(self.agents[idx].i, self.agents[idx].j)

        self._move()

        # After performing the action
        self._add_food()

        self.agents = self.grid.get_entities(EntityTypes.agent)

        self._remove_dead_agents()
        self._remove_old_pheromones()

        # Observations are a dict mapping agent names to their obs. Not all
        # agents need to be present in the dict in each time step.
        self.state = {}
        rewards = {}
        dones = {}
        truncates = {}
        infos = {}

        #print("After: ", [agent.id for agent in self.agents])
        for idx, agent in enumerate(self.agents):
            self.state[agent.id] = self._get_agent_obs(idx)
            rewards[agent.id], dones[agent.id], truncates[agent.id], infos[agent.id] = self._get_rewards(idx)

        if not (False in dones.values()):
            dones["__all__"] = True
        else:
            dones["__all__"] = False

        truncates["__all__"] = False

        self._update_agent_state()

        #if dones["__all__"] == True:
        #print("AFTER2: ", dones)
        return self.state, rewards, dones, truncates, infos
    
    def render(self):
        if self.render_mode == "human":
            return self.SimRenderer.render(self.agents, self.grid, fps=self.metadata_env["render_fps"])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def agent_key(self, gene, agent_index, type: Union[Agent, Wolf] = Agent):
          type_str = "agent" if type == Agent else "wolf"
          return type_str + "-gene-" + str(gene) + "-num-" + str(agent_index + 1)
    
    def _get_rewards(self, idx):
        done = False

        nr_kin_alive = max(0, sum([1 for other_agent in self.agents if not other_agent.dead and
                                   self.agents[idx].gene == other_agent.gene]) - 1)
        alive_agents = sum([1 for other_agent in self.agents if not other_agent.dead])
        if self.agents[idx].dead:
            print("HERE")
            reward = -10 #(-1 * alive_agents) + nr_kin_alive
            done = True
        else:

            if alive_agents == 1:
                if self.agents[idx].has_eaten in [5, 10, -5]:
                    if (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.75: reward = self.agents[idx].has_eaten
                    elif (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.50: reward = 1.2 * self.agents[idx].has_eaten
                    elif (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.25: reward = 1.6 * self.agents[idx].has_eaten
                    elif (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.0: reward = 2 * self.agents[idx].has_eaten
                else:
                    reward = -0.1
                # reward = (self.agents[idx].health / self.agents[idx].max_health) + \
                #         (self.agents[idx].hunger / self.agents[idx].max_hunger)
                        #(self.agents[idx].thirst / self.agents[idx].max_thirst)

                if self.agents[idx].has_drunk:
                    if (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.75: reward += 0.5
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.50: reward += 0.8
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.25: reward += 1
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.0: reward += 2
                elif self.agents[idx].thirst == 0:
                    reward -= 0.5

            else:
                # Percentage of population with the same kin + cur health/max health
                if self.agents[idx].has_eaten in [5, 10, -5]:
                    if (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.75: reward = self.agents[idx].has_eaten
                    elif (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.50: reward = 1.2 * self.agents[idx].has_eaten
                    elif (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.25: reward = 1.6 * self.agents[idx].has_eaten
                    elif (self.agents[idx].hunger / self.agents[idx].max_hunger) >= 0.0: reward = 2 * self.agents[idx].has_eaten
                else:
                    reward = -0.1

                if self.agents[idx].has_drunk:
                    #print("WATER NICE")
                    if (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.75: reward += 0.5
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.50: reward += 0.8
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.25: reward += 1
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.0: reward += 2
                elif self.agents[idx].thirst == 0:
                    #print("NEED TO DRINK")
                    reward -= 0.5

                        #(self.agents[idx].thirst / self.agents[idx].max_thirst)
                #reward = (agent.health / agent.max_health) + (agent.hunger / agent.max_hunger) + (agent.thirst / agent.max_thirst)
                #reward = agent.has_eaten

            if self.agents[idx].killed:
                if self.agents[idx].inter_killed:
                    reward -= 0.5
                else:
                    reward += 4

            if self.agents[idx].miss_reproduced:
                reward -= 1
            elif self.agents[idx].birth_delay == 30:
                reward += 10

            if self.agents[idx].miss_attacked:
                reward -= 1
        if type(self.agents[idx]) == Wolf:
            print("WOLF: ", reward, self.agents[idx].health)

        #print("HMMMM: ", reward)

        return reward, done, False, {}

    def _get_agent_obs(self, agent_idx):
        # Get separate grid for separate entities
        get_gene = np.vectorize(lambda item: self._get_gene(item))
        gene_grid = get_gene(self.grid.grid)

        get_type = np.vectorize(lambda item: self._get_type(item))
        type_grid = get_type(self.grid.grid)

        get_food = np.vectorize(lambda item: self._get_food(item))
        food_grid = get_food(self.grid.grid)

        get_water = np.vectorize(lambda item: self._get_water(item))
        water_grid = get_water(self.grid.grid)

        get_trees = np.vectorize(lambda item: self._get_tree(item))
        tree_grid = get_trees(self.grid.grid)

        get_health = np.vectorize(
            lambda obj: obj.health / obj.max_health if obj.entity_type == EntityTypes.agent else -1)
        health_grid = get_health(self.grid.grid)

        #reproduced = self.agents[agent_idx].birth_delay == 30

        gene_observed = self.grid.get_surroundings(self.agents[agent_idx].i,self.agents[agent_idx].j, 3, gene_grid)
        gene_observed[(gene_observed != 0) & (gene_observed != self.agents[agent_idx].gene)] = -6.
        gene_observed[gene_observed == self.agents[agent_idx].gene] = 6.
        gene_observed = list(gene_observed.flatten())
        type_observed = list(self.grid.get_surroundings(self.agents[agent_idx].i, self.agents[agent_idx].j, 3, type_grid).flatten())
        food_observed = list(self.grid.get_surroundings(self.agents[agent_idx].i, self.agents[agent_idx].j, 3, food_grid).flatten())
        water_observed = list(self.grid.get_surroundings(self.agents[agent_idx].i, self.agents[agent_idx].j, 3, water_grid).flatten())
        tree_observed = list(self.grid.get_surroundings(self.agents[agent_idx].i, self.agents[agent_idx].j, 3, tree_grid).flatten())
        health_observed = list(self.grid.get_surroundings(self.agents[agent_idx].i, self.agents[agent_idx].j, 3, health_grid).flatten())
        
        observation = np.array(gene_observed +
                                type_observed +
                                food_observed +
                                health_observed +
                                water_observed +
                                tree_observed +
                                [self.agents[agent_idx].health / self.agents[agent_idx].max_health] +
                                [self.agents[agent_idx].thirst / self.agents[agent_idx].max_thirst] +
                                [self.agents[agent_idx].hunger / self.agents[agent_idx].max_hunger] +
                                [self.agents[agent_idx].birth_delay] +
                                [self.agents[agent_idx].i, self.agents[agent_idx].j])
        
        # Update their states
        self.agents[agent_idx].state_prime = observation

        if self.agents[agent_idx].age == 0:
            self.agents[agent_idx].state = observation

        return observation

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def _init_water(self):
        self.grid.set_water()

    def _init_consumables(self,
                        entity: Union[Type[Food], Type[Poison]],
                        probability: float):
        entity_type = entity([-1, -1]).entity_type
        for _ in range(self.size * self.size):
            if np.random.random() < probability:
                self.grid.set_random(entity, p=1)

    def _init_trees(self,
                    entity: Type[Tree],
                    probability: float):
        entity_type = entity([-1, -1]).entity_type
        for _ in range(self.size * self.size):
            if np.random.random() < probability:
                self.grid.set_random(entity, p=1)

    def _add_agent(self,
               coordinates: Collection[int] = None,
               brain: Brain = None,
               gene: int = None,
               type: Union[Type[Agent], Type[Wolf]] = Agent,
               random_loc: bool = False,
               p: float = 1.) -> Type[Entity] or None:
        self.agent_count += 1
        if random_loc:
            return self.grid.set_random(type, p=p, brain=brain, gene=gene)
        else:
            return self.grid.set_cell(coordinates[0], coordinates[1], type, brain=brain, gene=gene)
        

    def _add_food(self):
        if len(np.where(self.grid.get_grid() == EntityTypes.food)[0]) <= ((self.size * self.size) / 10):
            for _ in range(3):
                self.grid.set_random(Food, p=0.2)

        if len(np.where(self.grid.get_grid() == EntityTypes.poison)[0]) <= ((self.size * self.size) / 20):
            for _ in range(3):
                self.grid.set_random(Poison, p=0.2)

    def _move(self):
        impossible_coordinates = True
        while impossible_coordinates:

            impossible_coordinates = self._get_impossible_coordinates()
            for agent in self.agents:
                if (agent.new_i, agent.new_j) in impossible_coordinates:
                    agent.update_new_location(agent.i, agent.j)
        
        for agent in self.agents:
            if agent.action <= 3:
                self._eat(agent)
                self._drink(agent)
                self._update_agent_position(agent)

    def _drink(self, agent: Union[Type[Agent], Type[Wolf]]):
        if agent.i + 1 < self.grid.width and agent.i - 1 > -1 and agent.j + 1 < self.grid.height and agent.j - 1 > -1 and EntityTypes.water in [
            self.grid.grid[agent.i-1, agent.j].entity_type,
            self.grid.grid[agent.i, agent.j-1].entity_type,
            self.grid.grid[agent.i+1, agent.j].entity_type,
            self.grid.grid[agent.i, agent.j+1].entity_type]:

            agent.thirst = 200 if type(agent) == Wolf else 100
            agent.has_drunk = True
    
    def _eat(self, agent: Union[Type[Agent], Type[Wolf]]):
        if type(agent) == Agent:
            if self.grid.grid[agent.new_i, agent.new_j].entity_type == EntityTypes.food:
                agent.hunger = min(100, agent.hunger + 40)
                agent.hunger = max(0, agent.hunger)
                agent.has_eaten = 5

            elif self.grid.grid[agent.new_i, agent.new_j].entity_type == EntityTypes.corpse:
                agent.hunger = 100
                agent.has_eaten = 10

            elif self.grid.grid[agent.new_i, agent.new_j].entity_type == EntityTypes.poison:
                agent.hunger = min(100, agent.hunger - 10)
                agent.hunger = max(0, agent.hunger)
                agent.health = min(200, agent.health - 10)
                agent.has_eaten = -5
        elif type(agent) == Wolf:
            eaten_agents = self._get_eaten_agents(agent)

            if len(eaten_agents) > 0:
                # print("EATEN")
                agent.hunger = agent.max_hunger
                agent.has_eaten = 10

                for agent in eaten_agents:
                    agent.health = 0

            if self.grid.grid[agent.new_i, agent.new_j].entity_type == EntityTypes.corpse:
                agent.hunger = agent.max_hunger
                agent.has_eaten = 10

        # else: # hasn't eaten anything
        #     agent.has_eaten = -0.01

    def _update_agent_state(self):
        for agent in self.agents:
            agent.state = agent.state_prime
            # print("ID: ", agent.id, "health: ", agent.health, "hunger: ", agent.hunger, "thirst: ", agent.thirst)

            # Heal if not hungry and not thirsty
            if agent.hunger >= 50 and agent.thirst >= 50:
                agent.health = min(agent.max_health, agent.health + 5)
            
            # Heal if not hungry only
            elif agent.hunger >= 50:
                agent.health = min(agent.max_health, agent.health + 2)

            # Heal if not thirsty only
            elif agent.thirst >= 50:
                agent.health = min(agent.max_health, agent.health + 2)

            # Take damage if hungry
            if agent.hunger == 0:
                agent.health = max(0, agent.health - 5)

            if agent.thirst == 0:
                agent.health = max(0, agent.health - 5)

            # The agent must get hungry and thirsty after so much running! Phew...
            agent.hunger = max(0, agent.hunger - 2)
            agent.thirst = max(0, agent.thirst - 1)

            # Reduce the birth delay
            agent.birth_delay = max(0, agent.birth_delay - 1)

            # Reset stats
            agent.has_eaten = 0
            agent.has_drunk = False
            agent.miss_attacked = False
            agent.miss_reproduced = False
            agent.inter_killed = False
            agent.miss_killed = False
            agent.killed = 0

    def _update_agent_position(self, agent: Agent):
        # Needed because otherwise sometimes some agents are deleted accidentally
        if not (type(self.grid.grid[agent.i, agent.j]) == Agent and \
                self.grid.grid[agent.i, agent.j] != agent):
            # print(type(self.grid.grid[agent.i, agent.j]))
            # if type(self.grid.grid[agent.i, agent.j]) == Wolf and self.grid.grid[agent.i, agent.j] != agent:
            #     print("YESSSSSS")
            self.grid.grid[agent.i, agent.j] = Pheromone((agent.i, agent.j))
        self.grid.grid[agent.new_i, agent.new_j] = agent
        agent.move()

    def _crossover_weights(self, weights1, weights2):
        new_weights = collections.OrderedDict()

        for key in weights1.keys():
            # Randomly choose a crossover point
            crossover_point = np.random.choice(range(len(weights1[key])))
            # Create the new weight tensor
            new_weight = torch.cat((weights1[key][:crossover_point], weights2[key][crossover_point:]), dim=0)
            new_weights[key] = new_weight
            
        return new_weights

    def _crossover(self, agent1, agent2):
        if not isinstance(agent1.brain, RandBrain) and \
            not isinstance(agent2.brain, RandBrain):
            child_brain = DQNBrain() if isinstance(agent1.brain, DQNBrain) else DDQNBrain()
            weights1 = agent1.brain.agent.state_dict()
            weights2 = agent2.brain.agent.state_dict()
            # Apply crossover on the weights
            new_weights = self._crossover_weights(weights1, weights2)
            child_brain.agent.load_state_dict(new_weights)
            child_brain.target.load_state_dict(new_weights)
            # Mutate the genes of the child brain
            child_brain.mutate()
            return child_brain
        else:
            return RandBrain()
        
    def _remove_dead_agents(self):
        for agent in self.agents:
            if agent.health <= 0:# or agent.age == agent.max_age:
                agent.dead = True
                # Update statistics
                self.dead_agents.append(agent)
                self.dead_scores.append(agent.age)

                self.agents.remove(agent)

                self.grid.grid[agent.i, agent.j] = Corpse((agent.i, agent.j))

    def _remove_old_pheromones(self):
        pheromones = self.grid.get_entities(EntityTypes.pheromone)
        for pheromon in pheromones:
            pheromon.lasting_time -= 1
            if pheromon.lasting_time <= 0:
                self.grid.grid[pheromon.i, pheromon.j] = Empty((pheromon.i, pheromon.j))

    def _get_empty_cells_surroundings(self, agent: Agent) -> List[Union[int, int]] or List[None]:
        observation = self.grid.get_surroundings(agent.i, agent.j, 3)

        loc = []
        for i in range(len(observation)):
            for j in range(len(observation[1])):
                if observation[i][j].entity_type == 0:
                    loc.append((i, j))

        coordinates = []

        for i_local, j_local in loc:
            diff_x = i_local - 3
            diff_y = j_local - 3

            # Get coordinates if within normal range
            global_x = agent.i + diff_x
            global_y = agent.j + diff_y

            # Get coordinates if through wall (left vs right)
            if global_y < 0:
                global_y = self.size + global_y
            elif global_y >= self.size:
                global_y = global_y - self.size

            # Get coordinates if through wall (up vs down)
            if global_x < 0:
                global_x = self.size + global_x
            elif global_x >= self.size:
                global_x = global_x - self.size

            coordinates.append([global_x, global_y])

        return coordinates
    
    def _get_impossible_coordinates(self):
        # If they want in to go into the water, update their next coordinates
        for agent in self.agents:
            if (agent.new_i, agent.new_j) in self.grid.water_coordinates or \
                (agent.new_i, agent.new_j) in self.grid.tree_coordinates:
                agent.update_new_location(agent.i, agent.j)

        new_coordinates = [(agent.new_i, agent.new_j) for agent in self.agents]
        if new_coordinates:
            unq, count = np.unique(new_coordinates, axis=0, return_counts=True)
            impossible_coordinates = [(coordinate[0], coordinate[1]) for coordinate in unq[count > 1]]
            return impossible_coordinates
        else:
            return []

    def _get_adjacent_agent(self, agent1: Agent, radius = 2) -> Agent:
        agents = [(agent.i, agent.j, agent) for agent in self.agents]

        for i in range(len(agents)):
            x, y, agent2 = agents[i]
            if agent1 != agent2 and abs(x - agent1.i) <= radius and abs(y - agent1.j) <= radius:
                return agent2
            
    def _get_eaten_agents(self, agent1: Agent, radius: int = 1) -> List[Agent]:
        agents = [(agent.i, agent.j, agent) for agent in self.agents]
        eaten_agents = []

        for i in range(len(agents)):
            x, y, agent2 = agents[i]
            if agent1 != agent2 and type(agent2) == Agent and abs(x - agent1.i) <= radius and abs(y - agent1.j) <= radius:
                eaten_agents.append(agent2)
        
        return eaten_agents

    def _get_gene(self, obj: Entity or Agent) -> int:
        if obj.entity_type == EntityTypes.agent and not obj.dead:
            return obj.gene
        else:
            return 0.
    
    def _get_type(self, obj: Entity or Agent or Wolf) -> int:
        if type(obj) == Agent and not obj.dead:
            return 4.
        elif type(obj) == Wolf and not obj.dead:
            return -4.
        else:
            return 0.
    
    def _get_food(self, obj: Entity or Agent) -> float:
        if obj.entity_type == EntityTypes.food:
            return .5
        elif obj.entity_type == EntityTypes.poison:
            return -1.
        elif obj.entity_type == EntityTypes.corpse:
            return 3.
        else:
            return 0.

    def _get_water(self, obj: Entity or Agent) -> float:
        if obj.entity_type == EntityTypes.water:
            return 2.
        else:
            return 0.
    
    def _get_tree(self, obj: Entity or Agent) -> float:
        if obj.entity_type == EntityTypes.tree:
            return -3.
        else:
            return 0.

    # Add new agents if the population is too small. The brain of the new agent will be copied from 
    # the current oldest agent
    def _generate_agent(self):

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

                    new_agent = self._add_agent(random_loc=True, brain=new_brain, gene=selected_agent.gene, type=type(selected_agent))
                else:
                    new_agent = self._add_agent(random_loc=True, brain=RandBrain(), gene=selected_agent.gene, type=type(selected_agent))
                
                new_agent.id = self.agent_key(selected_agent.gene, self.agent_count, type=type(selected_agent))
                self.agents.append(new_agent)
            
            # Reset metrics
            self.dead_agents = []
            self.dead_scores = []
            self.lives = []
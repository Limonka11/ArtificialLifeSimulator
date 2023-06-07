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

class Survival_Map(MultiAgentEnv):
    metadata_env = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, config: dict = {}):
        self.size = config.get("size", 50)
        self.square_size = 15
        self.grid = Grid(self.size, self.size)
        self.SimRenderer = SimRenderer(self.size, self.size, self.square_size)
        self.window_size = 512

        self.agents = []
        self.num_brains = config.get("brains", 5)
        self.max_agents = config.get("max_agents", 20)

        # State of each agent
        self.state = {}
        self.agent_count = 0

        self.mutation_stats_bunnies = []
        self.mutation_stats_wolfs = []

        # Metrics
        self.dead_agents = []
        self.dead_scores = []
        self.lives = []
        self.gens = [[],[],[]]
        self.gens_wolfs = [[],[],[]]
        self.max_avg_age = 0
        self.max_old_age = 0
        self.episode_num = 0

        # Change those with Gym Spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-100, high=300, shape=(59,), dtype=np.float32)

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
        self.episode_num += 1

        self.agents = []
        self.state = {}
        infos = {}


        self.grid = Grid(self.size, self.size)

        # Reset water
        self._init_water()
        
        # Reset Consumables
        self._init_consumables(Food, probability = 0.05)
        self._init_consumables(Poison, probability = 0.05)
        self._init_trees(Tree, probability=0.02)

        # Reset agents
        self.agents = []

        print("HELP: ", self.mutation_stats_bunnies)
        if self.episode_num % 500 == 0:
            # For more metrics regarding agents
            self.show_stats(self.episode_num // 10)

        # Genetical algorithm
        if len(self.mutation_stats_bunnies) != 0 and len(self.mutation_stats_wolfs) != 0:
            # Metrics
            avg_attack_bun = np.average([agent[0] for agent in self.mutation_stats_bunnies])
            avg_armor_bun = np.average([agent[1] for agent in self.mutation_stats_bunnies])
            avg_agility_bun = np.average([agent[2] for agent in self.mutation_stats_bunnies])

            avg_attack_wolf = np.average([agent[0] for agent in self.mutation_stats_wolfs])
            avg_armor_wolf = np.average([agent[1] for agent in self.mutation_stats_wolfs])
            avg_agility_wolf = np.average([agent[2] for agent in self.mutation_stats_wolfs])

            #print(avg_attack_bun, avg_armor_bun, avg_agility_bun, avg_attack_wolf, avg_armor_wolf, avg_agility_wolf)

            self.gens[0].append(avg_attack_bun)
            self.gens[1].append(avg_armor_bun)
            self.gens[2].append(avg_agility_bun)
    
            self.gens_wolfs[0].append(avg_attack_wolf)
            self.gens_wolfs[1].append(avg_armor_wolf)
            self.gens_wolfs[2].append(avg_agility_wolf)


            total_age_bunnies = sum([stats[3] for stats in self.mutation_stats_bunnies])
            select_probs_bunnies = [agent[3] / total_age_bunnies for agent in self.mutation_stats_bunnies]

            total_age_wolfs= sum([stats[3] for stats in self.mutation_stats_wolfs])
            select_probs_wolfs = [wolf[3] / total_age_wolfs for wolf in self.mutation_stats_wolfs]
            #print("PROBS: ", select_probs_wolfs, "WOLFS: ", self.mutation_stats_wolfs)
            #print("PROBS: ", select_probs_bunnies, "BUNN: ", self.mutation_stats_bunnies)

        for idx in range(self.num_brains):

            if len(self.mutation_stats_bunnies) != 0 and len(self.mutation_stats_wolfs) != 0:
                if(self.agent_count+1) % 4 == 0:
                    #print("WOLF")
                    # Use a weighted random choice to select an agent, with higher chances for agents with higher scores
                    selected_wolf_idx = np.random.choice(len(self.mutation_stats_wolfs), p=select_probs_wolfs)
                    selected_stats = self.mutation_stats_wolfs[selected_wolf_idx]
                else:
                    #print("BUNN")
                    selected_bunny_idx = np.random.choice(len(self.mutation_stats_bunnies), p=select_probs_bunnies)
                    selected_stats = self.mutation_stats_bunnies[selected_bunny_idx]

                #Mutate attack damage
                if np.random.random() < 0.02: # mutation_rate
                    #noise = selected_stats[0] * 0.01 # mutation_std = 0.01
                    if np.random.random() <= 0.5:
                        selected_stats[0] += 1
                    else:
                        selected_stats[0] -= 1

                # Mutate armor
                if np.random.random() < 0.02: # mutation_rate
                    #noise = selected_stats[1] * 0.1 # mutation_std = 0.01
                    if np.random.random() <= 0.5:
                        selected_stats[1] += 1
                    else:
                        selected_stats[1] -= 1

                # Mutate agility
                if np.random.random() < 0.02: # mutation_rate
                    noise = selected_stats[2] * 0.05 # mutation_std = 0.01
                    if np.random.random() <= 0.5:
                        selected_stats[2] += noise
                    else:
                        selected_stats[2] -= noise
                
                cur_agent = self._add_agent(random_loc=True,
                                            gene=(self.agent_count+1) % 2 + 10,
                                            type=Agent if (self.agent_count+1) % 4 != 0 else Wolf,
                                            attack_damage=selected_stats[0],
                                            armor=selected_stats[1],
                                            agility=selected_stats[2])
            else:
                if self.episode_num > 3:
                    raise Exception("We shoudn't have to regenerate agents!")
                # Create a new agent and add in the agent list
                cur_agent = self._add_agent(random_loc=True,
                                            gene=(self.agent_count+1) % 2 + 10,
                                            type=Agent if (self.agent_count+1) % 4 != 0 else Wolf)
                
            cur_agent.id = self.agent_key(cur_agent.gene, self.agent_count, type(cur_agent))
            #print("KEY: ", cur_agent.id)
            self.agents.append(cur_agent)
            self.state[cur_agent.id] = self._get_agent_obs(idx)
            infos[cur_agent.id] = {"armor": cur_agent.armor, "attack_damage": cur_agent.attack_damage, "agility": cur_agent.agility}

        # Metrics
        self.dead_agents = []
        self.dead_scores = []
        self.lives = []
        #self.gens = [[],[],[]]
        self.max_avg_age = 0
        self.max_old_age = 0

        # Mutational statistics
        self.mutation_stats_bunnies = []
        self.mutation_stats_wolfs = []

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

                    if (self.agents[idx].can_breed()) and (len(self.agents) <= self.max_agents) and adjacent_agent and (adjacent_agent.can_breed()) and (type(self.agents[idx]) is type(adjacent_agent)):
                
                        new_ad, new_armor, new_agility = self._crossover(self.agents[idx], adjacent_agent)

                        # Update the reward value
                        self.agents[idx].has_reproduced = self.agents[idx].libido
                        adjacent_agent.has_reproduced = adjacent_agent.libido

                        # Set birth delay
                        self.agents[idx].birth_delay = 10
                        adjacent_agent.birth_delay = 10

                        # Update libido

                        # Add offspring close to parent
                        coordinates = self._get_empty_cells_surroundings(self.agents[idx])

                        if coordinates:
                            new_agent = self._add_agent(coordinates=coordinates[random.randint(0, len(coordinates) - 1)],
                                                        gene=self.agents[idx].gene,
                                                        type=type(self.agents[idx]),
                                                        attack_damage=new_ad,
                                                        armor=new_armor,
                                                        agility=new_agility)
                        else:
                            new_agent = self._add_agent(random_loc=True,
                                                        gene=self.agents[idx].gene,
                                                        type=type(self.agents[idx]),
                                                        attack_damage=new_ad,
                                                        armor=new_armor,
                                                        agility=new_agility)

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
                    dodged = target.is_attacked(self.agents[idx].attack_damage)
                    self.agents[idx].execute_attack(target.armor, target.health, dodged)

                    if type(self.agents[idx]) == Agent:
                        if target.gene == self.agents[idx].gene:
                            self.agents[idx].inter_attacked = True
                        else:
                            self.agents[idx].inter_attacked = False

                    # Not needed
                    elif type(self.agents[idx]) == Wolf:
                        self.agents[idx].has_eaten = 10

                if target.entity_type == EntityTypes.wolf:
                    if type(self.agents[idx]) == Agent:
                        self.agents[idx].health = 0

                    elif type(self.agents[idx]) == Wolf:
                        dodged = target.is_attacked(self.agents[idx].attack_damage)
                        self.agents[idx].execute_attack(target.armor, target.health, dodged)
                        
                        if target.gene == self.agents[idx].gene:
                            self.agents[idx].inter_attacked = True
                        else:
                            self.agents[idx].inter_attacked = False
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
            reward = -10
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
                    if (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.75: reward += 0
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

                # TODO: Increase the rewards for drinking
                if self.agents[idx].has_drunk:
                    #print("WATER NICE")
                    if (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.75: reward += 0
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.50: reward += 0.8
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.25: reward += 1
                    elif (self.agents[idx].thirst / self.agents[idx].max_thirst) >= 0.0: reward += 2
                elif self.agents[idx].thirst == 0:
                    #print("NEED TO DRINK")
                    reward -= 0.5

                        #(self.agents[idx].thirst / self.agents[idx].max_thirst)
                #reward = (agent.health / agent.max_health) + (agent.hunger / agent.max_hunger) + (agent.thirst / agent.max_thirst)
                #reward = agent.has_eaten

            if self.agents[idx].inter_attacked:
                reward -= 0.5
            elif self.agents[idx].killed:
                reward += 4

            # if self.agents[idx].miss_reproduced:
            #     reward -= 1
            
            if self.agents[idx].has_reproduced:
                reward += (2*self.agents[idx].has_reproduced)

            # if self.agents[idx].miss_attacked:
            #     reward -= 1
        # if type(self.agents[idx]) == Wolf:
        #     print("WOLF: ", reward, self.agents[idx].health)

        #print("HMMMM: ", reward)

        return reward, done, False, {}

    def _get_agent_obs(self, agent_idx):
        # Get separate grid for separate entities
        get_preprocessed_obs = np.vectorize(lambda item: self._get_preprocessed_obs(item))
        preprocessed_obs = get_preprocessed_obs(self.grid.grid)

        vicinity_observed = self.grid.get_surroundings(self.agents[agent_idx].i, self.agents[agent_idx].j, 3, preprocessed_obs)
        for index, x in np.ndenumerate(vicinity_observed):
            if x == -4.:
                if index[0]-1 >= 0:
                    if vicinity_observed[index[0]-1][index[1]] == 0.:
                        vicinity_observed[index[0]-1][index[1]] = -2
                    if index[1]-1 >= 0 and vicinity_observed[index[0]-1][index[1] -1] == 0.:
                        vicinity_observed[index[0]-1][index[1]-1] = -2
                    if index[1]+1 < vicinity_observed[1].size and vicinity_observed[index[0]-1][index[1] + 1] == 0.:
                        vicinity_observed[index[0]-1][index[1]+1] = -2
                #print(vicinity_observed[0].size)
                if index[0]+1 < vicinity_observed[0].size:
                    if vicinity_observed[index[0]+1][index[1]] == 0.:
                        vicinity_observed[index[0]+1][index[1]] = -2
                    if index[1]-1 >= 0 and vicinity_observed[index[0]+1][index[1]-1] == 0:
                        vicinity_observed[index[0]+1][index[1]-1] = -2
                    if index[1]+1 < vicinity_observed[1].size and vicinity_observed[index[0]+1][index[1]+1] == 0.:
                        vicinity_observed[index[0]+1][index[1]+1] = -2
                if index[1]+1 < vicinity_observed[1].size and vicinity_observed[index[0]][index[1]+1] == 0.:
                    vicinity_observed[index[0]][index[1]+1] = -2
                if index[1]-1 >= 0 and vicinity_observed[index[0]][index[1]-1] == 0.:
                    vicinity_observed[index[0]][index[1]-1] = -2
                    
        vicinity_flat = list(vicinity_observed.flatten())
        
        observation = np.array(vicinity_flat +
                                [self.agents[agent_idx].health / self.agents[agent_idx].max_health] +
                                [self.agents[agent_idx].thirst / self.agents[agent_idx].max_thirst] +
                                [self.agents[agent_idx].hunger / self.agents[agent_idx].max_hunger] +
                                [self.agents[agent_idx].birth_delay, self.agents[agent_idx].has_reproduced] +
                                [self.agents[agent_idx].i, self.agents[agent_idx].j] +
                                [self.agents[agent_idx].armor, self.agents[agent_idx].agility, self.agents[agent_idx].attack_damage])
        
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
               gene: int = None,
               type: Union[Type[Agent], Type[Wolf]] = Agent,
               attack_damage = 200,
               armor = 0,
               agility = 0.05,
               random_loc: bool = False,
               p: float = 1.) -> Type[Entity] or None:
        self.agent_count += 1
        if random_loc:
            return self.grid.set_random(type, p=p, gene=gene, attack_damage=attack_damage, armor=armor, agility=agility)
        else:
            return self.grid.set_cell(coordinates[0], coordinates[1], type, gene=gene, attack_damage=attack_damage, armor=armor, agility=agility)
        

    def _add_food(self):
        if len(np.where(self.grid.get_grid() == EntityTypes.food)[0]) <= ((self.size * self.size) // 15):
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

            for prey in eaten_agents:
                dodged = prey.is_attacked(agent.attack_damage)
                successful = agent.execute_attack(prey.armor, prey.health, dodged)
                
                if successful:
                    agent.hunger = agent.max_hunger
                    agent.has_eaten = 10

                    prey.health = 0

            # if len(eaten_agents) > 0:
            #     # print("EATEN")
            #     agent.hunger = agent.max_hunger
            #     agent.has_eaten = 10

            #     for agent in eaten_agents:
            #         agent.health = 0

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
            agent.has_reproduced = 0
            agent.inter_attacked = False
            agent.miss_killed = False
            agent.killed = 0

    def _update_agent_position(self, agent: Agent):
        # Needed because otherwise sometimes some agents are deleted accidentally
        if not (self.grid.grid[agent.i, agent.j].entity_type == EntityTypes.agent and \
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
        new_ad = (agent1.attack_damage + agent2.attack_damage)/2
        new_armor = (agent1.armor + agent2.armor)/2
        new_agility = (agent1.agility + agent2.agility)/2
        return new_ad, new_armor, new_agility
        
    def _remove_dead_agents(self):
        for agent in self.agents:
            if agent.health <= 0:# or agent.age == agent.max_age:
                agent.dead = True
                # Update statistics
                self.dead_agents.append(agent)
                self.dead_scores.append(agent.age)
                if type(agent) == Wolf:
                    self.mutation_stats_wolfs.append([agent.attack_damage, agent.armor, agent.agility, agent.age])
                else:
                    self.mutation_stats_bunnies.append([agent.attack_damage, agent.armor, agent.agility, agent.age])

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

    def _get_preprocessed_obs(self, obj) -> int:
        if type(obj) == Agent and not obj.dead:
            return 4.
        elif type(obj) == Wolf and not obj.dead:
            return -4.
        elif obj.entity_type == EntityTypes.food:
            return .5
        elif obj.entity_type == EntityTypes.poison:
            return -1.
        elif obj.entity_type == EntityTypes.corpse:
            return 3.
        elif obj.entity_type == EntityTypes.water:
            return 2.
        elif obj.entity_type == EntityTypes.tree:
            return -3.
        else:
            return 0.

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



    def show_stats(self, number):
        gens = range(len(self.gens[0]))
        avg_attack_bun = self.gens[0]
        avg_armor_bun = self.gens[1]
        avg_agility_bun = self.gens[2]

        plt.figure(1)
        plt.plot(gens, avg_attack_bun)
        plt.plot(gens, self.gens_wolfs[0])
        plt.title("Attack Damage")
        plt.legend(['Avg Attack Bunnies', 'Avg Attack Wolfs'], loc='upper left')
        plt.savefig("metrics/AD_{}.png".format(number))
        plt.close()

        plt.figure(2)
        plt.plot(gens, avg_armor_bun)
        plt.plot(gens, self.gens_wolfs[1])
        plt.title("Armor")
        plt.legend(['Avg Armor Bunnies', 'Avg Armor Wolfs'], loc='upper left')
        plt.savefig("metrics/armor_{}.png".format(number))
        plt.close()

        plt.figure(3)
        plt.plot(gens, avg_agility_bun)
        plt.plot(gens, self.gens_wolfs[2])
        plt.title("Agility")
        plt.legend(['Avg Agility Bunnies', 'Avg Agility wolfs'], loc='upper left')
        plt.savefig("metrics/agility_{}.png".format(number))
        plt.close()
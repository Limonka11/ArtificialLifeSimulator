import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Any
from Brain import Brain
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDQNBrain(Brain):
    def __init__(self, state_size=200, action_size=9, num_eps_explore=2000, update_nn_freq=300, train_freq=30,
                 learning_rate=1e-3, batch_size=64, capacity=10000, discount_rate=0.95, load_model=False, training=True):
        super().__init__(state_size, action_size, "DDQN")

        self.target = DuelingDDQN(state_size, action_size)
        self.agent = DuelingDDQN(state_size, action_size)

        # The eval and the target networks need to be the same in the beginning
        self.agent.load_state_dict(self.target.state_dict())

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.buffer = PriorityReplayBuffer(capacity)

        self.num_eps_explore = num_eps_explore
        self.update_nn_freq = update_nn_freq
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.decay = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.n_epi = 0
        self.training = training
        
        if not self.training:
            self.epsilon = 0

        if load_model:
            checkpoint = torch.load(load_model)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.agent.eval()

            self.target.load_state_dict(checkpoint['model_state_dict'])
            self.target.eval()

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def act(self, state, n_epi):
        if self.training:
            if n_epi > self.n_epi:
                if self.epsilon > self.epsilon_min:
                    self.epsilon = self.epsilon * self.decay
                self.n_epi = n_epi

        action = self.agent.act(torch.FloatTensor(np.expand_dims(state, 0)), self.epsilon)
        return action

    def train(self):
        observations, actions, rewards, next_observations, dones, indices, weights = self.buffer.sample(self.batch_size)
        
        observations = torch.FloatTensor(observations)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_observations = torch.FloatTensor(next_observations)
        dones = torch.FloatTensor(dones)

        q_values = self.agent.forward(observations)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target.forward(next_observations)
        next_q_value = next_q_values.max(1)[0].detach()
        expected_q_value = rewards + self.discount_rate * (1 - dones) * next_q_value

        loss = self.loss_fn(q_value, expected_q_value)

        # Update memory
        priorities = torch.abs(next_q_value - q_value).detach().numpy()
        self.buffer.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, age, dead, action, state, reward, state_prime, done, n_epi):
        # Add experience to the memory
        self.buffer.add(state, action, reward, state_prime, done)

        # If the episodes for exploring are over start training
        if n_epi > self.num_eps_explore:
            if age % self.train_freq == 0 or dead:
                self.train()

            if n_epi % self.update_nn_freq == 0:
                self.target.load_state_dict(self.agent.state_dict())

    def mutate(self):
        weights = self.agent.state_dict()
        for name, param in weights.items():
            if np.random.random() < 0.02: # mutation_rate
                noise = torch.randn(param.shape) * 0.01 # mutation_std = 0.01
                weights[name] += noise
        self.agent.load_state_dict(weights)
        self.target.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.target.state_dict(), path)

class PriorityReplayBuffer(object):
    def __init__(self, capacity, alpha=.6, beta=.4, beta_increment=1000):
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.idx = 0
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

    def add(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        max_prior = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append([observation, action, reward, next_observation, done])
        else:
            self.memory[self.idx] = [observation, action, reward, next_observation, done]
        
        self.priorities[self.idx] = max_prior

        # Move to the next free memory slot. If we are over the capacity start from the beginning
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[: len(self.memory)]
        else:
            probs = self.priorities
        
        # Calculate probabilities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        weights = (len(self.memory) * probs[indices]) ** (- self.beta)

        if self.beta < 1:
            self.beta += self.beta_increment

        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        observation, action, reward, next_observation, done = zip(* samples)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done, indices, weights

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority

class DuelingDDQN(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(DuelingDDQN, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.fc = nn.Linear(self.observation_dim, 128)

        self.advantage_stream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.value_stream = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, observation):
        features = self.fc(observation)
        advantage = self.advantage_stream(features)
        value = self.value_stream(features)
        return advantage + (value - advantage.mean())

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action
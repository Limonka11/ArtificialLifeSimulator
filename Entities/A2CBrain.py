import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Any
from Brain import Brain


class DDQNBrain(Brain):
    """ Prioritized Experience Replay Dueling Double Deep Q Network
    Parameters:
    -----------
    input_dim : int, default 153
        The input dimension
    output_dim : int, default 8
        The output dimension
    exploration : int, default 1000
        The number of epochs to explore the environment before learning
    soft_update_freq : int, default 200
        The frequency at which to align the target and eval nets
    train_freq : int, default 20
        The frequency at which to train the agent
    learning_rate : float, default 1e-3
        Learning rate
    batch_size : int
        The number of training samples to work through before the model's internal parameters are updated.
    gamma : float, default 0.98
        Discount factor. How far out should rewards in the future influence the policy?
    capacity : int, default 10_000
        Capacity of replay buffer
    load_model : str, default False
        Path to an existing model
    training : bool, default True,
        Whether to continue training or not
    """
    def __init__(self, input_dim=3271, output_dim=4, exploration=1000, soft_update_freq=200, train_freq=20,
                 learning_rate=1e-3, batch_size=64, capacity=10000, gamma=0.99, load_model=False, training=True):
        super().__init__(input_dim, output_dim, "DDQN")
        self.target = DuelingDDQN(input_dim, output_dim)
        self.agent = DuelingDDQN(input_dim, output_dim)

        # The eval and the target networks need to be the same in the beginning
        self.agent.load_state_dict(self.target.state_dict())

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.buffer = PrioritizedReplayBuffer(capacity)
        self.loss_fn = nn.MSELoss()

        self.exploration = exploration
        self.soft_update_freq = soft_update_freq
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_epi = 0
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.decay = 0.99

        self.training = training
        if not self.training:
            self.epsilon = 0

        if load_model:
            self.agent.load_state_dict(torch.load(load_model))
            self.agent.eval()

            if self.training:
                self.target.load_state_dict(torch.load(load_model))
                self.target.eval()
                self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)

    def act(self, state, n_epi):
        if self.training:
            if n_epi > self.n_epi:
                if self.epsilon > self.epsilon_min:
                    self.epsilon = self.epsilon * self.decay
                self.n_epi = n_epi

        action = self.agent.act(torch.FloatTensor(np.expand_dims(state, 0)), self.epsilon)
        return action

    def memorize(self, obs, action, reward, next_obs, done):
        self.buffer.store(obs, action, reward, next_obs, done)

    def train(self):
        observation, action, reward, next_observation, done, indices, weights = self.buffer.sample(self.batch_size)

        observation = torch.FloatTensor(observation)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done)

        q_values = self.agent.forward(observation)
        next_q_values = self.target.forward(next_observation)
        next_q_value = next_q_values.max(1)[0].detach()
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value

        loss = self.loss_fn(q_value, expected_q_value)
        priorities = torch.abs(next_q_value - q_value).detach().numpy()
        self.buffer.update_priorities(indices, priorities)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, age, dead, action, state, reward, state_prime, done, n_epi):
        self.memorize(state, action, reward, state_prime, done)

        if n_epi > self.exploration:
            if age % self.train_freq == 0 or dead:
                self.train()

            if n_epi % self.soft_update_freq == 0:
                self.target.load_state_dict(self.agent.state_dict())

    def apply_gaussian_noise(self):
        with torch.no_grad():
            self.target.fc.weight.add_(torch.randn(self.target.fc.weight.size()))
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


class PrioritizedReplayBuffer(object):
    def __init__(self, capacity, alpha=.6, beta=.4, beta_increment=1000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.memory = []
        self.priorities = np.zeros([self.capacity], dtype=np.float32)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        max_prior = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append([observation, action, reward, next_observation, done])
        else:
            self.memory[self.pos] = [observation, action, reward, next_observation, done]
        self.priorities[self.pos] = max_prior
        self.pos += 1
        self.pos = self.pos % self.capacity

    def sample(self, batch_size):
        if len(self.memory) < self.capacity:
            probs = self.priorities[: len(self.memory)]
        else:
            probs = self.priorities
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        weights = (len(self.memory) * probs[indices]) ** (- self.beta)
        if self.beta < 1:
            self.beta += self.beta_increment
        weights = weights / np.max(weights)
        weights = np.array(weights, dtype=np.float32)

        observation, action, reward, next_observation, done = zip(* samples)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)


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
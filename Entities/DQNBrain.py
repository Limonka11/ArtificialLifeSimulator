import collections
import random

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from Brain import Brain

# Hyperparameters
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class DQNBrain(Brain):
    def __init__(self, input_dim=3271, output_dim=4, learning_rate=0.001, train_freq=20,
                 load_model=False, training=True):
        super().__init__(input_dim, output_dim, "DQN")
        self.agent = Qnet(input_dim)
        self.target = Qnet(input_dim)
        self.target.load_state_dict(self.agent.state_dict())
        self.memory = ReplayBuffer()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.epsilon = 0.50
        self.train_freq = train_freq
        self.decay_rate = 0.001

        self.training = training
        if not self.training:
            self.epsilon = 0

        if load_model:
            self.agent.load_state_dict(torch.load(load_model))
            self.agent.eval()

    def act(self, state, n_epi):

        if self.training:
            if n_epi % 30 == 0:
                self.epsilon = max(0.1, self.epsilon / ( 1 + self.decay_rate * n_epi))

        return self.agent.sample_action(state, self.epsilon)

    def memorize(self, s, a, r, s_prime, done):
        if done:
            done_mask = 0.0
        else:
            done_mask = 1.0
        self.memory.put((s, a, r, s_prime, done_mask))

    def train(self):
        if self.memory.size() > 1000:
            train(self.agent, self.target, self.memory, self.optimizer)
        self.target.load_state_dict(self.agent.state_dict())

    def learn(self, age, dead, action, state, reward, state_prime, done):
        self.memorize(state, action, reward, state_prime, done)

        if age % self.train_freq == 0 or dead:
            self.train()

    def mutate(self):
        weights = self.agent.state_dict()
        for name, param in weights.items():
            if np.random.random() < 0.02: # mutation_rate
                noise = torch.randn(param.shape) * 0.01 # mutation_std = 0.01
                weights[name] += noise
        self.agent.load_state_dict(weights)
        self.target.load_state_dict(weights)


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self, input_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float()
        out = self.forward(obs)
        p = random.random()
        if p < epsilon:
            return random.randint(0, 3)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer):
    for i in range(5):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
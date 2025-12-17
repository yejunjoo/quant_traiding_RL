# dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQN:
    def __init__(self, obs_dim, action_dim=21, lr=5e-4, gamma=0.99, buffer_size=50000, batch_size=64):
        self.device = "cuda"

        # Action Definition (Discretization)
        # -1: Strong Sell; sell 50
        # -0.95 : sell 45
        # unit: 5
        self.n_actions = action_dim
        self.action_values = np.linspace(-1.0, 1.0, self.n_actions)
        self.action_map = {idx: self.action_values[idx] for idx in range(self.n_actions)}

        # Networks
        self.q_net = QNetwork(obs_dim, self.n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_every = 1000
        self.step_count = 0

    def act(self, obs, eval_mode=False):
        # Epsilon-Greedy Strategy
        if not eval_mode and random.random() < self.epsilon:
            # exploration
            action_idx = random.randint(0, self.n_actions - 1)
        else:
            # exploitation
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(obs)
                action_idx = q_values.argmax().item()

        continuous_action = np.array([self.action_map[action_idx]], dtype=np.float32)
        return continuous_action, action_idx

    def step(self, obs, action_idx, reward, next_obs, done):
        # Store transition in buffer
        self.buffer.push(obs, action_idx, reward, next_obs, done)
        self.step_count += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Target Network Hard Update
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.q_net(states).gather(1, actions)

        # Target Q values (Double DQN logic usually, but here simple DQN for brevity)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        return loss.item()
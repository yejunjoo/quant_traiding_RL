# ppo.py

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Normal



class PPO:
    def __init__(self, actor, critic, rollout_storage, obs_shape, action_shape):
        self.device = "cuda"
        self.actor = actor
        self.critic = critic
        self.storage = Storage(num_transition=rollout_storage, obs_shape=obs_shape, action_shape=action_shape)
        self.batch_sampler = self.storage.mini_batch_generator_shuffle

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=5e-4)
        self.multi_epoch_learning = 4
        self.num_mini_batches = 64
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gamma = 0.998
        self.lamda = 0.95

    def act(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.action, self.action_log_prob = self.actor.sample(obs)
        return self.action.cpu().numpy()

    def step(self, obs, reward, truncated, terminated):
        self.storage.add_transitions(obs, self.action, self.action_log_prob, reward, truncated, terminated)

    def update(self, obs):
        last_value = self.critic.predict(obs)
        self.storage.compute_returns(last_value, self.critic, self.gamma, self.lamda)
        loss = self._train_step()
        self.storage.clear()
        return loss

    def _train_step(self):
        loss_sum = 0

        for multi_epoch in range(self.multi_epoch_learning):
            for obs_batch, action_batch, action_log_prob_batch, advantage_batch, value_batch, return_batch \
                    in self.batch_sampler(self.num_mini_batches):


                new_action_log_prob_batch, entropy_batch = self.actor.evaluate(obs_batch, action_batch)

                ratio = torch.exp(new_action_log_prob_batch - action_log_prob_batch)
                surrogate_unclipped = (-1.0) * advantage_batch * ratio
                surrogate_clipped = (-1.0) * advantage_batch * torch.clamp(ratio, 1.0 -self.clip_param, 1.0 +self.clip_param)
                surrogate_loss = torch.max(surrogate_unclipped, surrogate_clipped).mean()

                new_value_batch = self.critic.evaluate(obs_batch)
                value_clipped = value_batch + (new_value_batch - value_batch).clamp(-self.clip_param, self.clip_param)
                value_loss_unclipped = (new_value_batch - return_batch).pow(2)
                value_loss_clipped =(value_clipped - return_batch).pow(2)
                value_loss =torch.max(value_loss_clipped, value_loss_unclipped).mean()

                loss = surrogate_loss \
                       + self.value_loss_coef * value_loss \
                       - self.entropy_coef *entropy_batch.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

        return loss




def init_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()

        # obs * hidden dim * hidden dim * action dim
        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mu = self.net(obs)
        return mu

    def sample(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        mu = self.forward(obs)

        clamped_log_std = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(clamped_log_std)
        # std = torch.exp(self.log_std)

        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate(self, obs, action):
        mu = self.forward(obs)


        clamped_log_std = torch.clamp(self.log_std, min=-20, max=2)
        std = torch.exp(clamped_log_std)
        # std = torch.exp(self.log_std)

        dist = Normal(mu, std)
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action_log_prob, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            init_layer(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def predict(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            value = self.net(obs)
        return value

    def evaluate(self, obs):
        return self.net(obs)


class Storage:
    def __init__(self, num_transition, obs_shape, action_shape):
        self.device = 'cuda'
        self.num_transition = num_transition

        self.step = 0
        self.obss = np.zeros([num_transition, obs_shape], dtype=np.float32)
        self.actions = np.zeros([num_transition, action_shape], dtype=np.float32)
        self.action_log_probs = np.zeros([num_transition, 1], dtype=np.float32)
        self.rewards = np.zeros([num_transition], dtype=np.float32)
        self.truncateds = np.zeros([num_transition], dtype=np.bool)  # 0: False
        self.terminateds = np.zeros([num_transition], dtype=np.bool) # 0: False

        self.values = np.zeros([num_transition], dtype=np.float32)
        self.advantages = np.zeros([num_transition], dtype=np.float32)
        self.returns = np.zeros([num_transition], dtype=np.float32)


    def add_transitions(self, obs, action, action_log_prob, reward, truncated, terminated):
        self.obss[self.step] = obs
        self.actions[self.step] = action.detach().cpu().numpy().flatten()
        self.action_log_probs[self.step] = action_log_prob.detach().cpu().numpy()
        self.rewards[self.step] = reward
        self.truncateds[self.step] = truncated
        self.terminateds[self.step] = terminated

        self.step += 1
        return

    def clear(self):
        self.step = 0

    def compute_returns(self, last_value, critic, gamma, lamda):
        obss_tensor = torch.tensor(self.obss, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            values_tensor = critic.predict(obss_tensor)
            self.values = values_tensor.detach().cpu().numpy().flatten()

            # self.values = critic.predict(self.obss)
            last_value = last_value.detach().cpu().numpy().flatten()[0]

        advantage = 0   # Last step Advantage

        for step in reversed(range(self.num_transition)):
            if step == (self.num_transition - 1):
                # last step
                next_value = last_value
            else:
                next_value = self.values[step+1]

            is_termination = 1.0 - self.terminateds[step]   # 0: False
            delta = self.rewards[step] + (gamma *next_value *is_termination) - self.values[step]
            advantage = delta + (gamma *lamda *advantage *is_termination)
            self.advantages[step] = advantage

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


    def mini_batch_generator_shuffle(self, num_mini_batches):

        batch_size = self.num_transition // num_mini_batches
        indices = np.random.permutation(self.num_transition)

        for i in range(num_mini_batches):

            start = i * batch_size
            end = start + batch_size
            batch_idx = indices[start:end]

            obs_batch = torch.tensor(self.obss[batch_idx], dtype=torch.float32).to(self.device)
            action_batch = torch.tensor(self.actions[batch_idx], dtype=torch.float32).to(self.device)
            action_log_prob_batch = torch.tensor(self.action_log_probs[batch_idx], dtype=torch.float32).unsqueeze(-1).to(self.device)
            advantage_batch = torch.tensor(self.advantages[batch_idx], dtype=torch.float32).unsqueeze(-1).to(self.device)
            return_batch = torch.tensor(self.returns[batch_idx], dtype=torch.float32).unsqueeze(-1).to(self.device)
            value_batch = torch.tensor(self.values[batch_idx], dtype=torch.float32).to(self.device)
            if value_batch.ndim == 1:
                value_batch = value_batch.unsqueeze(-1)

            yield obs_batch, action_batch, action_log_prob_batch, advantage_batch, value_batch, return_batch
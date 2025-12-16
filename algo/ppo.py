import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class PPO:
    def __init__(self, actor, critic, rollout_storage, obs_shape, action_shape):
        self.actor = actor
        self.critic = critic
        self.storage = Storage(num_transition=rollout_storage, obs_shape=obs_shape, action_shape=action_shape) # ??
        self.batch_sampler = BatchSampler() # ??

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=5e-4)
        self.multi_epoch_learning = 4
        self.num_mini_batches = 64
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.0     # why zero?
        self.max_grad_norm = 0.5
        self.gamma = 0.998
        self.lamda = 0.95

    def act(self, obs):
        self.action, self.action_log_prob = self.actor.sample(obs)
        return self.action

    def step(self, obs, reward, truncated, terminated):
        self.storage.add_transitions(obs, self.action, self.action_log_prob, reward, truncated, terminated)

    def update(self, obs):
        last_value = self.critic.predict(obs)
        self.storage.compute_returns(last_value, self.critic, self.gamma, self.lamda)
        loss = self._train_step()
        self.storage.clear()

    def _train_step(self):
        loss = None

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
                value_loss_unclipped = (value_batch - return_batch).pow(2)
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

class Actor:
    def __init__(self, architecture, distribution):
        self.architecture = architecture
        self.distribution = distribution

    def sample(self, obs):

        return action, action_log_prob

    def evaluate(self, obs, action):

        return action_log_prob, entropy

class Critic:
    def __init__(self):

    def predict(self, obs):
        return value

    def evaluate(self, obs):
        return value

class Storage:
    def __init__(self, num_transition, obs_shape, action_shape):
        self.num_transition = num_transition

        self.step = 0
        self.obss = np.zeros([num_transition, *obs_shape], dtype=np.float32)
        self.actions = np.zeros([num_transition, *action_shape], dtype=np.float32)
        self.action_log_probs = np.zeros([num_transition], dtype=np.float32)
        self.rewards = np.zeros([num_transition], dtype=np.float32)
        self.truncateds = np.zeros([num_transition], dtype=np.bool)  # 0: False
        self.terminateds = np.zeros([num_transition], dtype=np.bool) # 0: False

        self.values = np.zeros([num_transition], dtype=np.float32)
        self.advantages = np.zeros([num_transition], dtype=np.float32)
        self.returns = np.zeros([num_transition], dtype=np.float32)


    def add_transitions(self, obs, action, action_log_prob, reward, truncated, terminated):
        self.obss[self.step] = obs
        self.actions[self.step] = action
        self.action_log_probs[self.step] = action_log_prob
        self.rewards[self.step] = reward
        self.truncateds[self.step] = truncated
        self.terminateds[self.step] = terminated

        self.step += 1
        return

    def clear(self):
        self.step = 0

    def compute_returns(self, last_value, critic, gamma, lamda):
        with torch.no_grad():
            self.values = critic.predict(self.obss)

        advantage = 0   # Last step Advantage

        for step in reversed(range(self.num_transition)):
            if step == (self.num_transition - 1):
                # last step
                next_value = last_value
            else:
                next_value = self.values[step+1]

            is_termination = 1.0 - self.terminateds[step]   # 0: False
            delta = self.rewards[step] + (gamma *next_value *is_termination) - self.values[step]
            advantage = delta + (gamma *lamda *advantage)
            self.advantages[step] = advantage

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
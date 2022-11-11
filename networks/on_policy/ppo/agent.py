import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from encoder_init import EncodeState
from networks.on_policy.ppo.ppo import ActorCritic
from parameters import LATENT_DIM
from parameters import PPO_CHECKPOINT_DIR


device = torch.device("cpu")

class Buffer:
    def __init__(self):
         # Batch data
        self.observation = []     # batch observations
        self.actions = []         # batch actions
        self.log_probs = []       # log probs of each action
        self.rewards = []         # batch rewards
        self.dones = []

    def clear(self):
        del self.observation[:]     # batch observations
        del self.actions[:]         # batch actions
        del self.log_probs[:]       # log probs of each action
        del self.rewards[:]         # batch rewards
        del self.dones[:]

class PPOAgent(object):
    def __init__(self,action_std_init=0.4):
        
        #self.env = env
        self.obs_dim = 100
        self.action_dim = 2
        self.clip = 0.1
        self.gamma = 0.95
        self.n_updates_per_iteration = 10
        self.lr = 2e-4
        self.encode = EncodeState(LATENT_DIM)
        self.memory = Buffer()
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, action_std_init)
        #self.actor_optim = Adam(self.policy.actor.parameters(), lr=self.lr)
        #self.critic_optim = Adam(self.policy.critic.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr}])

        self.old_policy = ActorCritic(self.obs_dim, self.action_dim, action_std_init)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()


    def get_action(self, obs):

        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.tensor(obs, dtype=torch.float)
            action, logprob = self.old_policy.get_action_and_log_prob(obs.to(device))

        self.memory.observation.append(obs.to(device))
        self.memory.actions.append(action)
        self.memory.log_probs.append(logprob)

        return action.detach().cpu().numpy().flatten()

    def learn(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.observation, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.n_updates_per_iteration):

            # Evaluating old actions and values
            logprobs, values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match values tensor dimensions with rewards tensor
            values = torch.squeeze(values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.memory.clear()
    
    def save(self):
        torch.save(self.old_policy.state_dict(), PPO_CHECKPOINT_DIR+"/ppo_policy.pth")
   
    def load(self):
        self.old_policy.load_state_dict(torch.load(PPO_CHECKPOINT_DIR+"/ppo_policy.pth"))
        self.policy.load_state_dict(torch.load(PPO_CHECKPOINT_DIR+"/ppo_policy.pth"))
            
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent(object):
    def __init__(self, env):
        
        self.env = env
        self.obs_dim = 100
        self.action_dim = 2
        self.timesteps_per_batch = 12288
        self.max_timesteps_per_episode = 3048
        self.clip = 0.1
        self.gamma = 0.95
        self.n_updates_per_iteration = 10
        self.lr = 2e-4
        self.encode = EncodeState(LATENT_DIM)

        self.policy = ActorCritic(self.obs_dim, self.action_dim)
        self.actor_optim = Adam(self.policy.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.policy.critic.parameters(), lr=self.lr)
        #self.MseLoss = nn.MSELoss()


    def rollout(self):
        # Batch data
        batch_observation = []     # batch observations
        batch_actions = []         # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rewards = []         # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        episode_reward = []
        # Number of timesteps run so far this batch
        t = 0 
        while t < self.timesteps_per_batch:
        # Rewards this episode
        
            episode_reward = []

            obs = self.env.reset()
            obs = self.encode.process(obs)
            
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_observation.append(obs.cpu().numpy())
                action, log_prob = self.policy.get_action_and_log_prob(obs.cpu())
                obs, rew, done, _ = self.env.step(action)
                obs = self.encode.process(obs)

                # Collect reward, action, and log prob
                episode_reward.append(rew)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                
                if done:
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rewards.append(episode_reward)
        
        # Reshape data as tensors in the shape specified before returning
        batch_observation = torch.tensor(np.array(batch_observation), dtype=torch.float) #torch.stack(batch_observation)#
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float)
        batch_log_probs = torch.tensor(np.array(batch_log_probs), dtype=torch.float)
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rewards)
        # Return the batch data
        return batch_observation, batch_actions, batch_log_probs, batch_rtgs, batch_lens

    
    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(np.array(batch_rtgs), dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        # Calculate the log probabilities of batch actions using most 
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.policy.get_action(batch_obs)
        dist = MultivariateNormal(mean, self.policy.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        V = self.policy.get_value(batch_obs).squeeze()
        # Return predicted values V and log probs log_probs
        return V, log_probs


    def learn(self, total_timesteps):
        t_so_far = 0 # Timesteps simulated so far
        while t_so_far < total_timesteps:              # ALG STEP 2
            batch_observation, batch_actions, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            #print(len("batch len: ",batch_lens," sum(", np.sum(batch_lens), ") "))
            # Calculate how many timesteps we collected this batch   
            t_so_far += np.sum(batch_lens)
            
            # Calculate V_phi and pi_theta(a_t | s_t)    
            V, curr_log_probs = self.evaluate(batch_observation, batch_actions)
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()
            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            print(t_so_far)
            for _ in range(self.n_updates_per_iteration):
                print(_, end=" ")
                V , curr_log_probs = self.evaluate(batch_observation, batch_actions)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Calculate gradients and perform backward propagation for actor 
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()


                critic_loss = nn.MSELoss()(V, batch_rtgs)
                # Calculate gradients and perform backward propagation for critic network    
                self.critic_optim.zero_grad()    
                critic_loss.backward()    
                self.critic_optim.step()
    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path))
            
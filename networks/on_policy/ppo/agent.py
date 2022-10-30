import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from networks.on_policy.ppo.memory import PPOMemory
from networks.on_policy.ppo.ppo import ActorNetwork, CriticNetwork
from parameters import PPO_BATCH_SIZE, GAMMA, POLICY_CLIP, PPO_ACTOR, PPO_CRITIC


class PPOAgent(object):

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.gamma = GAMMA
        self.policy_clip = POLICY_CLIP
        self.batch_size = PPO_BATCH_SIZE
        self.N = 2048
        self.gae_lambda = 0.95
        self.n_epochs = 10

        self.actor = ActorNetwork(self.n_actions, PPO_ACTOR)
        self.critic = CriticNetwork(self.n_actions, PPO_CRITIC)
        self.memory = PPOMemory()
        
    def save_transition(self, visual_state , nav_data, action, probs, vals, reward, done):
        self.memory.save_memory((visual_state , nav_data), action, probs, vals, reward, done)

    def pick_action(self, visual_state, nav_data):
        visual_state = torch.tensor([visual_state], dtype=torch.float).to(self.actor.device)
        nav_data = torch.tensor([nav_data], dtype=torch.float).to(self.actor.device)

        dist = self.actor(visual_state, nav_data)
        value = self.critic(visual_state, nav_data)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action= torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def save_model(self):
        print('...saving model...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_model(self):
        print('...loading model...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def train(self):

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr , reward_arr, done_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k]+ self.gamma*values[k+1]*(1-int(done_arr[k]))-values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns -critic_value)**2
                critic_loss = critic_loss.mean()


                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()    


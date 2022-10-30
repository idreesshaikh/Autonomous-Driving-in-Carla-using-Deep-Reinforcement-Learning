import time
from unicodedata import name
import numpy as np
import torch
import torch.nn.functional as F
from networks.off_policy.sac.sac import ActorNetwork, CriticNetwork, ValueNetwork
from networks.off_policy.replay_buffer import ReplayBuffer
from parameters import *


class SACAgent(object):

    def __init__(self, n_actions):
        self.gamma = GAMMA
        self.reward_scale = 2
        self.tau = TAU
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE, (128,128, 3), (4, 1), n_actions)
        
        self.actor = ActorNetwork(n_actions, ACTOR_SAC)
        self.critic_1 = CriticNetwork(n_actions, CRITIC_1_SAC)
        self.critic_2 = CriticNetwork(n_actions, CRITIC_2_SAC)
        self.value = ValueNetwork(n_actions, VALUE_SAC)
        self.target_value = ValueNetwork(n_actions, TARGET_VALUE_SAC)
        
        self.scale = self.reward_scale
        self.update_network_parameters(tau=1)
        
    def save_transition(self, visual_state, action, nav_data,  reward, new_visual_state, new_nav_data, done):
        self.replay_buffer.save_transition(visual_state, action, nav_data, reward, new_visual_state, new_nav_data, done)

    def pick_action(self, visual_state, nav_data):
        visual_state = torch.tensor(visual_state, dtype=torch.float).to(self.actor.device)
        nav_data = torch.tensor(nav_data, dtype=torch.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(visual_state, nav_data, reparameterize=False)
        return actions.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
        self.target_value.load_state_dict(value_state_dict)


    def save_model(self):
        print('...saving model...')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_model(self):
        print('...loading model...')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def train(self):
        if self.replay_buffer.counter < self.batch_size:
            return

        visual_state, action, nav_data, reward, new_visual_state, new_nav_data, done = self.replay_buffer.sample_buffer()


        visual_state = torch.tensor(visual_state).to(self.actor.device)
        action = torch.tensor(action).to(self.actor.device)
        nav_data = torch.tensor(nav_data).to(self.actor.device)
        reward = torch.tensor(reward).to(self.actor.device)
        new_visual_state = torch.tensor(new_visual_state).to(self.actor.device)
        new_nav_data = torch.tensor(new_nav_data).to(self.actor.device)
        done = torch.tensor(done).to(self.actor.device)

        value = self.value.forward(visual_state, nav_data).view(-1)
        value_ = self.target_value.forward(new_visual_state, new_nav_data).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(visual_state, nav_data, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(visual_state,nav_data, actions)
        q2_new_policy = self.critic_2.forward(visual_state,nav_data, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(visual_state, nav_data, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(visual_state,nav_data, actions)
        q2_new_policy = self.critic_2.forward(visual_state,nav_data, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward +self.gamma*value_
        q1_old_policy = self.critic_1.forward(visual_state, nav_data, action).view(-1)
        q2_old_policy = self.critic_2.forward(visual_state, nav_data, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

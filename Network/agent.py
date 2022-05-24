import random
import torch
import numpy as np
from Network.ddqn import DDQnetwork
from Network.replay_buffer import ReplayBuffer
from parameters import *


class Agent(object):

    def __init__(self, n_actions, double=True):
        self.gamma = GAMMA
        self.alpha = LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_end = EPSILON_END
        self.double = double
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.train_step = 0
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE, (128, 128, 3), (4), n_actions)
        self.q_network_eval = DDQnetwork(n_actions, MODEL_ONLINE)
        self.q_network_target = DDQnetwork(n_actions, MODEL_TARGET)

    def save_transition(self, visual_state, action, nav_data,  reward, new_visual_state, new_nav_data, done):
        self.replay_buffer.save_transition(visual_state, action, nav_data, reward, new_visual_state, new_nav_data, done)

    def pick_action(self, visual_state, nav_data):
        if np.random.random() > self.epsilon:
            visual_state = torch.tensor(visual_state, dtype=torch.float).to(self.q_network_eval.device)
            nav_data = torch.tensor([nav_data], dtype=torch.float).to(self.q_network_eval.device)
            _, advantage = self.q_network_eval.forward(visual_state, nav_data)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def decrese_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= EPSILON_DECREMENT
        else:
            self.epsilon = self.epsilon_end

    def save_model(self):
        self.q_network_eval.save_checkpoint()
        self.q_network_target.save_checkpoint()

    def load_model(self):
        self.q_network_eval.load_checkpoint()
        self.q_network_target.load_checkpoint()

    def train(self):
        if self.replay_buffer.counter < self.batch_size:
            return

        self.q_network_eval.optimizer.zero_grad()

        if self.train_step % REPLACE_NETWORK == 0:
            self.q_network_target.load_state_dict(self.q_network_eval.state_dict())

        visual_state, action, nav_data, reward, new_visual_state, new_nav_data, done = self.replay_buffer.sample_buffer()

        visual_state = torch.tensor(visual_state).to(self.q_network_eval.device)
        action = torch.tensor(action).to(self.q_network_eval.device)
        nav_data = torch.tensor(nav_data).to(self.q_network_eval.device)
        reward = torch.tensor(reward).to(self.q_network_eval.device)
        new_visual_state = torch.tensor(new_visual_state).to(self.q_network_eval.device)
        new_nav_data = torch.tensor(new_nav_data).to(self.q_network_eval.device)
        done = torch.tensor(done).to(self.q_network_eval.device)

        indexes = np.arange(self.batch_size, dtype=np.int64)

        if self.double:

            Vs, As = self.q_network_eval.forward(visual_state, nav_data)
            nVs, nAs = self.q_network_target.forward(new_visual_state, new_nav_data)

            Vs_eval, As_eval = self.q_network_eval.forward(new_visual_state, new_nav_data)

            q_pred = torch.add(Vs, (As - As.mean(dim=1, keepdim=True)))[indexes, action]
            q_next =  torch.add(nVs, (nAs - nAs.mean(dim=1, keepdim=True)))
            q_eval =  torch.add(Vs_eval, (As_eval - As_eval.mean(dim=1, keepdim=True)))

            max_actions = torch.argmax(q_eval, dim=1)
            q_next[done] = 0.0
 
            q_target = reward + self.gamma*q_next[indexes, max_actions]

        else:
            q_pred = self.q_network_eval.forward(visual_state, nav_data)[indexes, action]
            q_next = self.q_network_target.forward(new_visual_state, new_nav_data).max(dim=1)[0]

            q_next[done] = 0.0
            q_target = reward + self.gamma*q_next

        loss = self.q_network_eval.loss(q_target, q_pred).to(self.q_network_eval.device)
        loss.backward()
        self.q_network_eval.optimizer.step()
        self.train_step += 1
        self.decrese_epsilon()
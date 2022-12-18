import torch
import numpy as np
from encoder_init import EncodeState
from networks.off_policy.ddqn.dueling_dqn import DuelingDQnetwork
from networks.off_policy.replay_buffer import ReplayBuffer
from parameters import *


class DQNAgent(object):

    def __init__(self, n_actions):
        self.gamma = GAMMA
        self.alpha = DQN_LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_end = EPSILON_END
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.train_step = 0
        #self.encode = EncodeState(LATENT_DIM)
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE,100, n_actions)
        self.q_network_eval = DuelingDQnetwork(n_actions, MODEL_ONLINE)
        self.q_network_target = DuelingDQnetwork(n_actions, MODEL_TARGET)

    def save_transition(self, observation, action,  reward, new_observation, done):
        self.replay_buffer.save_transition(observation, action, reward, new_observation, done)

    def get_action(self, observation):
        if np.random.random() > self.epsilon:
            #observation = self.encode.process(observation)
            #observation = torch.tensor(observation, dtype=torch.float).to(self.q_network_eval.device)
            _, advantage = self.q_network_eval.forward(observation)
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

    def learn(self):
        if self.replay_buffer.counter < self.batch_size:
            return

        self.q_network_eval.optimizer.zero_grad()

        if self.train_step % REPLACE_NETWORK == 0:
            self.q_network_target.load_state_dict(self.q_network_eval.state_dict())

        observation, action, reward, new_observation, done = self.replay_buffer.sample_buffer()
 

        observation = observation.to(self.q_network_eval.device)
        action = action.to(self.q_network_eval.device)
        reward = reward.to(self.q_network_eval.device)
        new_observation = new_observation.to(self.q_network_eval.device)
        done = done.to(self.q_network_eval.device)


        Vs, As = self.q_network_eval.forward(observation)
        nVs, nAs = self.q_network_target.forward(new_observation)
        q_pred = torch.add(Vs, (As - As.mean(dim=1, keepdim=True))).gather(1,action.unsqueeze(-1)).squeeze(-1)
        q_next =  torch.add(nVs, (nAs - nAs.mean(dim=1, keepdim=True)))
        q_target = reward + self.gamma*torch.max(q_next, dim=1)[0].detach()
        q_next[done] = 0.0
        loss = self.q_network_eval.loss(q_target, q_pred).to(self.q_network_eval.device)
        loss.backward()
        self.q_network_eval.optimizer.step()
        self.train_step += 1
        self.decrese_epsilon()
import numpy as np
import torch
from parameters import BATCH_SIZE



class ReplayBuffer(object):
    def __init__(self, max_size, observation, n_actions):

        self.n_actions = n_actions
        self.buffer_size = max_size
        self.counter = 0
        self.state_memory = torch.zeros((self.buffer_size, observation), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.buffer_size, observation), dtype=torch.float32)
        self.action_memory = torch.zeros(self.buffer_size, dtype=torch.int64)
        self.reward_memory = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.terminal_memory = torch.zeros(self.buffer_size, dtype=torch.bool)

    def save_transition(self, state, action, reward, new_state, done):

        index = self.counter % self.buffer_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.counter += 1

    def sample_buffer(self):
        max = min(self.counter, self.buffer_size)
        batch = np.random.choice(max, BATCH_SIZE, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self. terminal_memory[batch]

        return states, actions, rewards, new_states, dones

    def __len__(self):
        return len(self.buffer_size)
